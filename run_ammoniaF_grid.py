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

### python run_ammoniaF_grid.py --visible --resolution 5 --case-timeout 10

## If Aspen is truly frozen and normal closing is not enough, use:

### python run_ammoniaF_grid.py --visible --resolution 5 --case-timeout 10 --force-kill-aspen-on-timeout


SCRIPT_DIR = Path(__file__).resolve().parent

SIMULATION_NAME = "ammoniaF"
SIMULATION_FILE = "ammoniaF.bkp"
INPUTS_CSV = SCRIPT_DIR / "ammoniaF_inputs.csv"
OUTPUTS_CSV = SCRIPT_DIR / "ammoniaF_outputs.csv"
CASE_GRID_CSV = SCRIPT_DIR / "ammoniaF_case_grid.csv"
RESULTS_CSV = SCRIPT_DIR / "ammoniaF_results_live.csv"

# Aspen stores NO3-IN on a MASS-FRAC basis, and direct writes to TOTFLOW
# trigger underspecification there. Use the total-flow input as a case-grid
# driver only, then write the corresponding component flows directly.
#
# H2FEED is different: FLOW\\MIXED\\H2 only changes the stream composition
# bookkeeping, not the actual H2 feedrate. The writable total-flow spec that
# updates the stream is TOTFLOW\\MIXED.
NO3_IN_TOTAL_FLOW_DRIVER = "Ft"
NO3_IN_LEGACY_COMPOSITION_INPUT = "Wno3"
H2FEED_TOTAL_FLOW_DRIVER = "Fh2"
H2FEED_TOTAL_FLOW_PATH = r"\Data\Streams\H2FEED\Input\TOTFLOW\MIXED"
COMPUTED_WNH3_NAME = "Wnh3"
COMPUTED_NH3_OUT_KGPH_NAME = "NH3_out_kgph"
S13_TOTAL_MASS_FLOW_PATH = r"\Data\Streams\S13\Output\RES_MASSFLOW"
S13_NH3_MASS_FLOW_PATH = r"\Data\Streams\S13\Output\MASSFLOW\MIXED\NH3"
CAL_PER_SEC_TO_MMKCAL_PER_HR = 3600.0 / 1_000_000_000.0
HEAT_DUTY_PATH_TOKENS = (
    r"\OUTPUT\QNET",
    r"\OUTPUT\QCALC",
    "DUTY",
)
DERIVED_STREAM_COMPONENTS = {
    NO3_IN_TOTAL_FLOW_DRIVER: {
        "NO3In_H2O_Flow": {
            "path": r"\Data\Streams\NO3-IN\Input\FLOW\MIXED\H2O",
            "mass_fraction": 0.89659,
        },
        "NO3In_CL_Flow": {
            "path": r"\Data\Streams\NO3-IN\Input\FLOW\MIXED\CL-",
            "mass_fraction": 0.0534,
        },
        "NO3In_NA_Flow": {
            "path": r"\Data\Streams\NO3-IN\Input\FLOW\MIXED\NA+",
            "mass_fraction": 0.03889,
        },
        "NO3In_NO3_Flow": {
            "path": r"\Data\Streams\NO3-IN\Input\FLOW\MIXED\NO3-",
            "mass_fraction": 0.00925,
        },
        "NO3In_SO4_Flow": {
            "path": r"\Data\Streams\NO3-IN\Input\FLOW\MIXED\SO4--",
            "mass_fraction": 0.00187,
        },
    },
}


def cal_per_sec_to_mmkcal_per_hr(value: object) -> float:
    return float(value) * CAL_PER_SEC_TO_MMKCAL_PER_HR


def is_heat_duty_output_path(path: str) -> bool:
    normalized_path = path.upper()
    return any(token in normalized_path for token in HEAT_DUTY_PATH_TOKENS)


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
        if item.name == H2FEED_TOTAL_FLOW_DRIVER and item.path != H2FEED_TOTAL_FLOW_PATH:
            print(
                "Using the writable H2FEED total-flow spec "
                f"'{H2FEED_TOTAL_FLOW_PATH}' for {H2FEED_TOTAL_FLOW_DRIVER} "
                f"instead of '{item.path}'."
            )
            updated_inputs.append(
                InputVariable(
                    name=item.name,
                    path=H2FEED_TOTAL_FLOW_PATH,
                    lower=item.lower,
                    upper=item.upper,
                )
            )
            continue

        updated_inputs.append(item)

    return updated_inputs


def read_required_node_value(aspen, path: str, output_name: str) -> float:
    node = aspen.Tree.FindNode(path)
    if node is None:
        raise ValueError(f"Aspen node not found: {path}")

    value = node.Value
    if value is None:
        raise ValueError(f"Cannot calculate {output_name}: value at {path} was blank.")

    return float(value)


def compute_s13_nh3_mass_flow(aspen, _result_row: dict[str, object]) -> float:
    return read_required_node_value(aspen, S13_NH3_MASS_FLOW_PATH, COMPUTED_NH3_OUT_KGPH_NAME)


def compute_s13_nh3_mass_fraction(aspen, _result_row: dict[str, object]) -> float:
    total_mass_flow = read_required_node_value(aspen, S13_TOTAL_MASS_FLOW_PATH, COMPUTED_WNH3_NAME)
    nh3_mass_flow = read_required_node_value(aspen, S13_NH3_MASS_FLOW_PATH, COMPUTED_WNH3_NAME)

    if total_mass_flow in (None, 0):
        raise ValueError(
            f"Cannot calculate {COMPUTED_WNH3_NAME}: total mass flow at {S13_TOTAL_MASS_FLOW_PATH} "
            f"was {total_mass_flow!r}."
        )
    if nh3_mass_flow is None:
        raise ValueError(
            f"Cannot calculate {COMPUTED_WNH3_NAME}: NH3 mass flow at {S13_NH3_MASS_FLOW_PATH} was blank."
        )

    return float(nh3_mass_flow) / float(total_mass_flow)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a full-factorial Aspen input grid for ammoniaF and run the simulation "
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
    heat_duty_output_transforms = {
        name: cal_per_sec_to_mmkcal_per_hr
        for name in heat_duty_output_names
    }
    if heat_duty_output_names:
        print(
            "Converting ammonia heat-duty outputs from cal/sec to MMkcal/hr before writing results: "
            + ", ".join(heat_duty_output_names)
        )

    input_names = {item.name for item in inputs}
    output_names = {item.name for item in outputs}
    derived_component_names = {
        derived_name
        for component_specs in DERIVED_STREAM_COMPONENTS.values()
        for derived_name in component_specs
    }

    if NO3_IN_LEGACY_COMPOSITION_INPUT in input_names:
        raise ValueError(
            f"Do not activate '{NO3_IN_LEGACY_COMPOSITION_INPUT}' in {INPUTS_CSV.name}. "
            f"When '{NO3_IN_TOTAL_FLOW_DRIVER}' is active, the NO3-IN composition is written automatically."
        )

    reserved_computed_output_names = {
        COMPUTED_WNH3_NAME,
        COMPUTED_NH3_OUT_KGPH_NAME,
    }

    reserved_input_name_collisions = sorted(reserved_computed_output_names & input_names)
    if reserved_input_name_collisions:
        raise ValueError(
            "The following names are reserved for automatically calculated outputs, "
            f"so do not use them as input names in {INPUTS_CSV.name}: "
            f"{', '.join(reserved_input_name_collisions)}"
        )

    derived_name_collisions = sorted(derived_component_names & input_names)
    if derived_name_collisions:
        raise ValueError(
            "The following names are reserved for automatically calculated NO3-IN component flows, "
            f"so do not activate them as independent inputs: {', '.join(derived_name_collisions)}"
        )

    reserved_output_name_collisions = sorted(reserved_computed_output_names & output_names)
    if reserved_output_name_collisions:
        raise ValueError(
            "Do not activate the following names in "
            f"{OUTPUTS_CSV.name}; they are calculated automatically from Aspen: "
            f"{', '.join(reserved_output_name_collisions)}"
        )

    output_name_collisions = sorted(derived_component_names & output_names)
    if output_name_collisions:
        raise ValueError(
            "The following names are reserved for automatically calculated NO3-IN component flows, "
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
        computed_output_fetchers={
            COMPUTED_WNH3_NAME: compute_s13_nh3_mass_fraction,
            COMPUTED_NH3_OUT_KGPH_NAME: compute_s13_nh3_mass_flow,
        },
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
