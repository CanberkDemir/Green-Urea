from __future__ import annotations

import argparse
from pathlib import Path

from aspen_grid_runner import (
    create_template_files,
    generate_case_grid_csv,
    InputVariable,
    load_input_variables,
    load_output_variables,
    run_case_grid,
    validate_name_collisions,
)

### python run_ureaF_grid.py --visible --resolution 5 --case-timeout 10

## If Aspen is truly frozen and normal closing is not enough, use:

### python run_ureaF_grid.py --visible --resolution 5 --case-timeout 10 --force-kill-aspen-on-timeout


SCRIPT_DIR = Path(__file__).resolve().parent

SIMULATION_NAME = "ureaF"
SIMULATION_FILE = "ureaF.bkp"
INPUTS_CSV = SCRIPT_DIR / "ureaF_inputs.csv"
OUTPUTS_CSV = SCRIPT_DIR / "ureaF_outputs.csv"
CASE_GRID_CSV = SCRIPT_DIR / "ureaF_case_grid.csv"
RESULTS_CSV = SCRIPT_DIR / "ureaF_results_live.csv"
NH3FEED_TOTAL_FLOW_DRIVER = "Fnh3"
NH3FEED_TOTAL_FLOW_PATH = r"\Data\Streams\NH3-IN\Input\TOTFLOW\MIXED"
UREA_OUT_TOTAL_MASS_FLOW_NAME = "Ft_UREA-OUT"
COMPUTED_WUREA_NAME = "Wurea"
DEF_OUT_STREAM_PATH = r"\Data\Streams\DEF-OUT\Output"
UREA_OUT_TOTAL_MASS_FLOW_PATH = rf"{DEF_OUT_STREAM_PATH}\RES_MASSFLOW"
UREA_OUT_UREA_MASS_FLOW_PATH = rf"{DEF_OUT_STREAM_PATH}\MASSFLOW\MIXED\UREA"
UREA_OUT_UREA_MASS_FRACTION_PATH = rf"{DEF_OUT_STREAM_PATH}\MASSFRAC\MIXED\UREA"
CAL_PER_SEC_TO_MMKCAL_PER_HR = 3600.0 / 1_000_000_000.0
HEAT_DUTY_PATH_TOKENS = (
    r"\OUTPUT\QNET",
    r"\OUTPUT\QCALC",
    "DUTY",
)
UREA_HEAT_DUTIES_ALREADY_MMKCAL_PER_HR = True

# Keep the urea runner flow aligned with the ammonia runner so the same resume,
# validation, derived-input, and computed-output hooks are available here too.
# Leave these mappings empty until you have urea-specific Aspen paths to add.
DERIVED_STREAM_COMPONENTS: dict[str, dict[str, dict[str, object]]] = {}


def cal_per_sec_to_mmkcal_per_hr(value: object) -> float:
    return float(value) * CAL_PER_SEC_TO_MMKCAL_PER_HR


def is_heat_duty_output_path(path: str) -> bool:
    normalized_path = path.upper()
    return any(token in normalized_path for token in HEAT_DUTY_PATH_TOKENS)


def read_optional_node_value(aspen, path: str):
    node = aspen.Tree.FindNode(path)
    if node is None:
        return None
    return node.Value


def compute_urea_out_mass_fraction(aspen, result_row: dict[str, object]) -> float:
    direct_mass_fraction = read_optional_node_value(aspen, UREA_OUT_UREA_MASS_FRACTION_PATH)
    if direct_mass_fraction not in ("", None):
        return float(direct_mass_fraction)

    total_mass_flow = result_row.get(UREA_OUT_TOTAL_MASS_FLOW_NAME)
    urea_mass_flow = read_optional_node_value(aspen, UREA_OUT_UREA_MASS_FLOW_PATH)

    if total_mass_flow in ("", None):
        total_mass_flow = read_optional_node_value(aspen, UREA_OUT_TOTAL_MASS_FLOW_PATH)

    if total_mass_flow in ("", None):
        raise ValueError(
            f"Cannot calculate {COMPUTED_WUREA_NAME}: Aspen did not return total mass flow from "
            f"'{UREA_OUT_TOTAL_MASS_FLOW_NAME}' or {UREA_OUT_TOTAL_MASS_FLOW_PATH}."
        )
    if urea_mass_flow in ("", None):
        raise ValueError(
            f"Cannot calculate {COMPUTED_WUREA_NAME}: Aspen did not return DEF-OUT UREA mass flow "
            f"from {UREA_OUT_UREA_MASS_FLOW_PATH}."
        )

    total_mass_flow_value = float(total_mass_flow)
    if total_mass_flow_value == 0.0:
        raise ValueError(
            f"Cannot calculate {COMPUTED_WUREA_NAME}: '{UREA_OUT_TOTAL_MASS_FLOW_NAME}' was 0."
        )

    return float(urea_mass_flow) / total_mass_flow_value


COMPUTED_OUTPUT_FETCHERS: dict[str, object] = {
    COMPUTED_WUREA_NAME: compute_urea_out_mass_fraction,
}


def add_derived_component_flows(case_values: dict[str, float]) -> dict[str, float]:
    updated_case_values = dict(case_values)

    for driver_name, component_specs in DERIVED_STREAM_COMPONENTS.items():
        if driver_name not in case_values:
            continue

        total_flow = float(case_values[driver_name])
        for derived_name, spec in component_specs.items():
            updated_case_values[derived_name] = total_flow * float(spec["mass_fraction"])

    return updated_case_values


def normalize_runtime_input_paths(inputs: list[InputVariable]) -> list[InputVariable]:
    updated_inputs: list[InputVariable] = []

    for item in inputs:
        if item.name == NH3FEED_TOTAL_FLOW_DRIVER and item.path != NH3FEED_TOTAL_FLOW_PATH:
            print(
                "Using the writable NH3-IN total-flow spec "
                f"'{NH3FEED_TOTAL_FLOW_PATH}' for {NH3FEED_TOTAL_FLOW_DRIVER} "
                f"instead of '{item.path}'."
            )
            updated_inputs.append(
                InputVariable(
                    name=item.name,
                    path=NH3FEED_TOTAL_FLOW_PATH,
                    lower=item.lower,
                    upper=item.upper,
                )
            )
            continue

        updated_inputs.append(item)

    return updated_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a full-factorial Aspen input grid for ureaF and run the simulation "
            "for every case while writing results live to CSV."
        )
    )
    parser.add_argument("--simulation-file", default=SIMULATION_FILE, help="Path to the Aspen simulation file.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=3,
        help="Number of evenly spaced values per active input variable.",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show the Aspen GUI while the automation is running.",
    )
    parser.add_argument(
        "--sleep-after-run",
        type=float,
        default=0.0,
        help="Optional wait time in seconds after each Aspen run.",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for one Aspen case before restarting Aspen.",
    )
    parser.add_argument(
        "--force-kill-aspen-on-timeout",
        action="store_true",
        help=(
            "After a case timeout, force-kill Aspen Plus processes before restarting. "
            "Use only when you do not have other Aspen work open."
        ),
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create the blank input/output template CSVs and stop.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Create the full input case-grid CSV and stop before running Aspen.",
    )
    parser.add_argument(
        "--overwrite-templates",
        action="store_true",
        help="Overwrite the template CSV files if they already exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    create_template_files(INPUTS_CSV, OUTPUTS_CSV, overwrite=args.overwrite_templates)
    print(f"Input template:  {INPUTS_CSV}")
    print(f"Output template: {OUTPUTS_CSV}")

    if args.init_only:
        print("Template creation complete. Fill in the CSV files, then rerun the script.")
        return

    inputs = load_input_variables(INPUTS_CSV)
    inputs = normalize_runtime_input_paths(inputs)
    outputs = load_output_variables(OUTPUTS_CSV)
    validate_name_collisions(inputs, outputs)
    heat_duty_output_names = [
        item.name
        for item in outputs
        if is_heat_duty_output_path(item.path)
    ]
    if UREA_HEAT_DUTIES_ALREADY_MMKCAL_PER_HR:
        heat_duty_output_transforms = None
        if heat_duty_output_names:
            print(
                "Leaving urea heat-duty outputs in their Aspen-reported basis because "
                "UREA_HEAT_DUTIES_ALREADY_MMKCAL_PER_HR=True: "
                + ", ".join(heat_duty_output_names)
            )
    else:
        heat_duty_output_transforms = {
            name: cal_per_sec_to_mmkcal_per_hr
            for name in heat_duty_output_names
        }
        if heat_duty_output_names:
            print(
                "Converting urea heat-duty outputs from cal/sec to MMkcal/hr before writing results: "
                + ", ".join(heat_duty_output_names)
            )

    input_names = {item.name for item in inputs}
    output_names = {item.name for item in outputs}
    derived_component_names = {
        derived_name
        for component_specs in DERIVED_STREAM_COMPONENTS.values()
        for derived_name in component_specs
    }
    computed_output_names = set(COMPUTED_OUTPUT_FETCHERS)

    derived_name_collisions = sorted(derived_component_names & input_names)
    if derived_name_collisions:
        raise ValueError(
            "The following names are reserved for automatically calculated urea component flows, "
            f"so do not activate them as independent inputs: {', '.join(derived_name_collisions)}"
        )

    computed_input_collisions = sorted(computed_output_names & input_names)
    if computed_input_collisions:
        raise ValueError(
            "The following names are reserved for automatically calculated urea outputs, "
            f"so do not use them as inputs: {', '.join(computed_input_collisions)}"
        )

    computed_output_collisions = sorted(computed_output_names & output_names)
    if computed_output_collisions:
        raise ValueError(
            "The following outputs are calculated automatically in the urea run script, "
            f"so keep them inactive in {OUTPUTS_CSV.name}: {', '.join(computed_output_collisions)}"
        )

    output_name_collisions = sorted(derived_component_names & output_names)
    if output_name_collisions:
        raise ValueError(
            "The following names are reserved for automatically calculated urea component flows, "
            f"so do not reuse them as outputs: {', '.join(output_name_collisions)}"
        )

    active_total_flow_drivers = [
        driver_name
        for driver_name in DERIVED_STREAM_COMPONENTS
        if driver_name in input_names
    ]
    derived_case_columns = [
        derived_name
        for driver_name in active_total_flow_drivers
        for derived_name in DERIVED_STREAM_COMPONENTS[driver_name]
    ] or None
    derived_input_path_by_case_column = (
        {
            derived_name: spec["path"]
            for driver_name in active_total_flow_drivers
            for derived_name, spec in DERIVED_STREAM_COMPONENTS[driver_name].items()
        }
        or None
    )
    case_only_input_names = active_total_flow_drivers or None

    total_cases = generate_case_grid_csv(
        inputs,
        args.resolution,
        CASE_GRID_CSV,
        extra_case_columns=derived_case_columns,
        case_row_transform=add_derived_component_flows if active_total_flow_drivers else None,
    )
    print(f"Saved {total_cases} cases to: {CASE_GRID_CSV}")

    if args.generate_only:
        print("Case grid generation complete. Aspen was not started.")
        return

    summary = run_case_grid(
        simulation_file=args.simulation_file,
        inputs=inputs,
        outputs=outputs,
        case_grid_csv=CASE_GRID_CSV,
        results_csv=RESULTS_CSV,
        total_cases=total_cases,
        visible=args.visible,
        sleep_after_run_sec=args.sleep_after_run,
        extra_input_path_by_case_column=derived_input_path_by_case_column,
        case_only_input_names=case_only_input_names,
        computed_output_fetchers=COMPUTED_OUTPUT_FETCHERS,
        output_value_transformers=heat_duty_output_transforms,
        resume_by_case_values=True,
        drop_failed_existing_results=True,
        case_timeout_sec=args.case_timeout,
        force_kill_aspen_on_timeout=args.force_kill_aspen_on_timeout,
    )

    print("")
    print(f"Finished {SIMULATION_NAME}")
    print(f"Successful cases: {summary.success_count}")
    print(f"Failed cases:     {summary.failure_count}")
    print(f"Results CSV:      {summary.results_csv}")


if __name__ == "__main__":
    main()
