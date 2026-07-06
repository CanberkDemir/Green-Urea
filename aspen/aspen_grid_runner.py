from __future__ import annotations

import csv
import itertools
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


INPUT_TEMPLATE_COLUMNS = ["name", "path", "lower", "upper", "active", "notes"]
OUTPUT_TEMPLATE_COLUMNS = ["name", "path", "active", "notes"]

RESERVED_RESULT_COLUMNS = {
    "case_id",
    "run_ok",
    "error_message",
    "elapsed_sec",
    "completed_at",
}


@dataclass(frozen=True)
class InputVariable:
    name: str
    path: str
    lower: float
    upper: float


@dataclass(frozen=True)
class OutputVariable:
    name: str
    path: str


@dataclass(frozen=True)
class RunSummary:
    total_cases: int
    success_count: int
    failure_count: int
    case_grid_csv: Path
    results_csv: Path


@dataclass(frozen=True)
class ResumeState:
    next_case_id: int
    last_success_case_id: int
    existing_rows_to_keep: List[Dict[str, str]]
    success_count: int
    failure_count: int


@dataclass(frozen=True)
class ValueResumeState:
    existing_rows_to_keep: List[Dict[str, str]]
    successful_case_keys: Set[Tuple[Tuple[str, str], ...]]
    success_count: int
    failure_count: int
    next_result_case_id: int
    dropped_failure_count: int


ComputedOutputFetcher = Callable[[object, Dict[str, object]], object]
OutputValueTransformer = Callable[[object], object]


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


def is_blank(value: object) -> bool:
    return str(value or "").strip() == ""


def parse_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text == "":
        return default
    return text in {"1", "true", "yes", "y", "t"}


def parse_float(value: object, field_name: str, row_number: int, csv_path: Path) -> float:
    if is_blank(value):
        raise ValueError(
            f"Column '{field_name}' is blank in row {row_number} of {csv_path}."
        )
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Column '{field_name}' in row {row_number} of {csv_path} must be numeric. "
            f"Received: {value!r}"
        ) from exc


def safe_float(value: object) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)


def parse_case_id(value: object, label: str) -> int:
    text = str(value or "").strip()
    if text == "":
        raise ValueError(f"{label} is blank.")

    try:
        numeric_value = float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer. Received: {value!r}") from exc

    if not numeric_value.is_integer():
        raise ValueError(f"{label} must be an integer. Received: {value!r}")

    case_id = int(numeric_value)
    if case_id < 1:
        raise ValueError(f"{label} must be at least 1. Received: {value!r}")

    return case_id


def normalize_case_key_value(value: object) -> str:
    text = str(value or "").strip()
    if text == "":
        return ""

    try:
        numeric_value = float(text)
    except ValueError:
        return text

    if not math.isfinite(numeric_value):
        return text

    return f"{numeric_value:.12g}"


def make_case_value_key(
    row: Dict[str, str],
    case_fieldnames: Sequence[str],
) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        (fieldname, normalize_case_key_value(row.get(fieldname, "")))
        for fieldname in case_fieldnames
        if fieldname != "case_id"
    )


def linspace(lower: float, upper: float, resolution: int) -> List[float]:
    if resolution < 1:
        raise ValueError("Resolution must be at least 1.")
    if resolution == 1:
        return [lower]
    step = (upper - lower) / (resolution - 1)
    values = [lower + step * index for index in range(resolution)]
    values[-1] = upper
    return values


def create_template_csv(path: Path, columns: Sequence[str], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    ensure_parent_directory(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()


def create_template_files(inputs_csv: Path, outputs_csv: Path, overwrite: bool = False) -> None:
    create_template_csv(inputs_csv, INPUT_TEMPLATE_COLUMNS, overwrite=overwrite)
    create_template_csv(outputs_csv, OUTPUT_TEMPLATE_COLUMNS, overwrite=overwrite)


def _load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is missing a header row.")

        normalized_fieldnames = [normalize_key(name) for name in reader.fieldnames]
        rows: List[Dict[str, str]] = []

        for raw_row in reader:
            normalized_row = {
                normalized_name: (raw_row.get(original_name) or "").strip()
                for original_name, normalized_name in zip(reader.fieldnames, normalized_fieldnames)
            }
            if all(is_blank(value) for value in normalized_row.values()):
                continue
            rows.append(normalized_row)

    return rows


def _assert_no_reserved_names(names: Iterable[str], csv_path: Path) -> None:
    for name in names:
        if name in RESERVED_RESULT_COLUMNS:
            raise ValueError(
                f"Variable name '{name}' in {csv_path} is reserved. "
                f"Choose a different name."
            )


def _assert_unique_names(names: Sequence[str], csv_path: Path, label: str) -> None:
    seen = set()
    duplicates = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate {label} names in {csv_path}: {duplicate_list}")


def load_input_variables(inputs_csv: Path) -> List[InputVariable]:
    rows = _load_csv_rows(inputs_csv)
    if not rows:
        raise ValueError(
            f"{inputs_csv} does not contain any active input variables yet. "
            f"Fill in at least one row with name, path, lower, and upper."
        )

    inputs: List[InputVariable] = []
    for row_number, row in enumerate(rows, start=2):
        if not parse_bool(row.get("active", ""), default=True):
            continue

        name = row.get("name", "")
        path = row.get("path", "")

        if is_blank(name):
            raise ValueError(f"Column 'name' is blank in row {row_number} of {inputs_csv}.")
        if is_blank(path):
            raise ValueError(f"Column 'path' is blank in row {row_number} of {inputs_csv}.")

        lower = parse_float(row.get("lower", ""), "lower", row_number, inputs_csv)
        upper = parse_float(row.get("upper", ""), "upper", row_number, inputs_csv)

        if upper < lower:
            raise ValueError(
                f"Upper bound must be greater than or equal to lower bound for input '{name}' "
                f"in row {row_number} of {inputs_csv}."
            )

        inputs.append(InputVariable(name=name, path=path, lower=lower, upper=upper))

    if not inputs:
        raise ValueError(
            f"{inputs_csv} contains rows, but none are active. Set 'active' to 1/true or leave it blank."
        )

    names = [item.name for item in inputs]
    _assert_no_reserved_names(names, inputs_csv)
    _assert_unique_names(names, inputs_csv, "input")
    return inputs


def load_output_variables(outputs_csv: Path) -> List[OutputVariable]:
    rows = _load_csv_rows(outputs_csv)
    if not rows:
        raise ValueError(
            f"{outputs_csv} does not contain any active output variables yet. "
            f"Fill in at least one row with name and path."
        )

    outputs: List[OutputVariable] = []
    for row_number, row in enumerate(rows, start=2):
        if not parse_bool(row.get("active", ""), default=True):
            continue

        name = row.get("name", "")
        path = row.get("path", "")

        if is_blank(name):
            raise ValueError(f"Column 'name' is blank in row {row_number} of {outputs_csv}.")
        if is_blank(path):
            raise ValueError(f"Column 'path' is blank in row {row_number} of {outputs_csv}.")

        outputs.append(OutputVariable(name=name, path=path))

    if not outputs:
        raise ValueError(
            f"{outputs_csv} contains rows, but none are active. Set 'active' to 1/true or leave it blank."
        )

    names = [item.name for item in outputs]
    _assert_no_reserved_names(names, outputs_csv)
    _assert_unique_names(names, outputs_csv, "output")
    return outputs


def validate_name_collisions(inputs: Sequence[InputVariable], outputs: Sequence[OutputVariable]) -> None:
    input_names = {item.name for item in inputs}
    output_names = {item.name for item in outputs}
    collisions = sorted(input_names & output_names)
    if collisions:
        collision_text = ", ".join(collisions)
        raise ValueError(
            "Input and output names must be distinct so the results CSV has unambiguous columns. "
            f"Conflicting names: {collision_text}"
        )


def generate_case_grid_csv(
    inputs: Sequence[InputVariable],
    resolution: int,
    case_grid_csv: Path,
    extra_case_columns: Optional[Sequence[str]] = None,
    case_row_transform: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
) -> int:
    if resolution < 2:
        raise ValueError("Resolution must be at least 2 so both lower and upper bounds are included.")
    if not inputs:
        raise ValueError("At least one input variable is required to generate a case grid.")

    ensure_parent_directory(case_grid_csv)

    input_names = [item.name for item in inputs]
    derived_names = list(extra_case_columns or [])
    value_grids = [linspace(item.lower, item.upper, resolution) for item in inputs]
    total_cases = int(math.prod(len(grid) for grid in value_grids))

    with case_grid_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", *input_names, *derived_names])
        writer.writeheader()

        for case_id, combination in enumerate(itertools.product(*value_grids), start=1):
            case_values = dict(zip(input_names, combination))
            transformed_values = dict(case_values)
            if case_row_transform is not None:
                transformed_values = case_row_transform(dict(case_values))

            row = {"case_id": case_id}
            row.update(case_values)

            for derived_name in derived_names:
                if derived_name not in transformed_values:
                    raise ValueError(
                        f"Derived case column '{derived_name}' was requested but not produced by case_row_transform."
                    )
                row[derived_name] = transformed_values[derived_name]

            writer.writerow(row)

    return total_cases


def resolve_simulation_path(simulation_file: str | Path) -> Path:
    if is_blank(simulation_file):
        raise ValueError(
            "Simulation file path is blank. Set SIMULATION_FILE in the script or pass --simulation-file."
        )

    model_path = Path(simulation_file).expanduser()
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Aspen simulation file not found: {model_path}")

    return model_path


def launch_aspen(simulation_file: str | Path, visible: bool = False):
    import win32com.client as win32

    model_path = resolve_simulation_path(simulation_file)
    extension = model_path.suffix.lower()

    aspen = win32.Dispatch("Apwn.Document")

    try:
        if extension in {".bkp", ".apw", ".apwz"}:
            aspen.InitFromArchive2(str(model_path))
        elif extension == ".apwn":
            aspen.InitFromFile2(str(model_path))
        else:
            raise ValueError(
                f"Unsupported Aspen file extension: {extension}. "
                "Recommended formats are .bkp, .apw, .apwz, or .apwn."
            )

        try:
            aspen.Visible = int(bool(visible))
        except Exception:
            pass

        # Leave Aspen dialogs enabled while debugging so Aspen can show its
        # own warning/error windows instead of hiding them from the runner.
        # try:
        #     aspen.SuppressDialogs = 1
        # except Exception:
        #     pass

        return aspen
    except Exception:
        try:
            aspen.Quit()
        except Exception:
            pass
        raise


def close_aspen(aspen) -> None:
    try:
        aspen.Close()
    except Exception:
        pass
    try:
        aspen.Quit()
    except Exception:
        pass


def force_kill_aspen_processes() -> None:
    for process_name in ("Apwn.exe", "AspenPlus.exe"):
        subprocess.run(
            ["taskkill", "/IM", process_name, "/F", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def find_node(aspen, path: str):
    node = aspen.Tree.FindNode(path)
    if node is None:
        raise ValueError(f"Aspen node not found: {path}")
    return node


def set_node_value(aspen, path: str, value: float) -> None:
    node = find_node(aspen, path)
    node.Value = value


def get_node_value(aspen, path: str):
    node = find_node(aspen, path)
    return node.Value


def wait_for_aspen_to_finish(aspen, timeout_sec: float = 600.0, poll_sec: float = 0.2) -> None:
    start_time = time.time()
    while True:
        try:
            is_running = bool(aspen.Engine.IsRunning)
        except Exception:
            return

        if not is_running:
            return

        if time.time() - start_time > timeout_sec:
            raise TimeoutError(f"Aspen did not finish running within {timeout_sec} seconds.")

        time.sleep(poll_sec)


def inspect_existing_results(
    results_csv: Path,
    expected_fieldnames: Sequence[str],
    total_cases: int,
) -> ResumeState:
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return ResumeState(
            next_case_id=1,
            last_success_case_id=0,
            existing_rows_to_keep=[],
            success_count=0,
            failure_count=0,
        )

    with results_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        existing_fieldnames = list(reader.fieldnames or [])
        if not existing_fieldnames:
            return ResumeState(
                next_case_id=1,
                last_success_case_id=0,
                existing_rows_to_keep=[],
                success_count=0,
                failure_count=0,
            )

        expected_fieldnames_list = list(expected_fieldnames)
        if existing_fieldnames != expected_fieldnames_list:
            raise ValueError(
                f"The existing results CSV header in {results_csv} does not match the current run configuration. "
                "Delete or rename the old results CSV before resuming."
            )

        latest_row_by_case_id: Dict[int, Dict[str, str]] = {}
        for raw_row in reader:
            row = {
                fieldname: "" if raw_row.get(fieldname) is None else str(raw_row.get(fieldname))
                for fieldname in existing_fieldnames
            }
            if all(is_blank(value) for value in row.values()):
                continue

            case_id = parse_case_id(
                row.get("case_id", ""),
                f"Column 'case_id' in {results_csv}",
            )
            if case_id > total_cases:
                raise ValueError(
                    f"Existing results CSV {results_csv} contains case_id {case_id}, "
                    f"but the current case grid only has {total_cases} cases."
                )
            latest_row_by_case_id[case_id] = row

    if not latest_row_by_case_id:
        return ResumeState(
            next_case_id=1,
            last_success_case_id=0,
            existing_rows_to_keep=[],
            success_count=0,
            failure_count=0,
        )

    last_success_case_id = 0
    for case_id, row in latest_row_by_case_id.items():
        if parse_bool(row.get("run_ok", ""), default=False):
            last_success_case_id = max(last_success_case_id, case_id)

    if last_success_case_id == 0:
        return ResumeState(
            next_case_id=1,
            last_success_case_id=0,
            existing_rows_to_keep=[],
            success_count=0,
            failure_count=0,
        )

    existing_rows_to_keep = [
        latest_row_by_case_id[case_id]
        for case_id in sorted(latest_row_by_case_id)
        if case_id <= last_success_case_id
    ]
    success_count = sum(
        1 for row in existing_rows_to_keep if parse_bool(row.get("run_ok", ""), default=False)
    )
    failure_count = len(existing_rows_to_keep) - success_count

    return ResumeState(
        next_case_id=min(last_success_case_id + 1, total_cases + 1),
        last_success_case_id=last_success_case_id,
        existing_rows_to_keep=existing_rows_to_keep,
        success_count=success_count,
        failure_count=failure_count,
    )


def inspect_existing_results_by_case_values(
    results_csv: Path,
    expected_fieldnames: Sequence[str],
    case_fieldnames: Sequence[str],
    drop_failed_rows: bool = False,
) -> ValueResumeState:
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return ValueResumeState(
            existing_rows_to_keep=[],
            successful_case_keys=set(),
            success_count=0,
            failure_count=0,
            next_result_case_id=1,
            dropped_failure_count=0,
        )

    with results_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        existing_fieldnames = list(reader.fieldnames or [])
        if not existing_fieldnames:
            return ValueResumeState(
                existing_rows_to_keep=[],
                successful_case_keys=set(),
                success_count=0,
                failure_count=0,
                next_result_case_id=1,
                dropped_failure_count=0,
            )

        expected_fieldnames_list = list(expected_fieldnames)
        if existing_fieldnames != expected_fieldnames_list:
            raise ValueError(
                f"The existing results CSV header in {results_csv} does not match the current run configuration. "
                "Delete or rename the old results CSV before accumulating more results."
            )

        existing_rows_to_keep: List[Dict[str, str]] = []
        successful_case_keys: Set[Tuple[Tuple[str, str], ...]] = set()
        max_case_id = 0
        success_count = 0
        failure_count = 0
        dropped_failure_count = 0

        for raw_row in reader:
            row = {
                fieldname: "" if raw_row.get(fieldname) is None else str(raw_row.get(fieldname))
                for fieldname in existing_fieldnames
            }
            if all(is_blank(value) for value in row.values()):
                continue

            case_id = parse_case_id(
                row.get("case_id", ""),
                f"Column 'case_id' in {results_csv}",
            )
            max_case_id = max(max_case_id, case_id)

            if parse_bool(row.get("run_ok", ""), default=False):
                existing_rows_to_keep.append(row)
                successful_case_keys.add(make_case_value_key(row, case_fieldnames))
                success_count += 1
            else:
                failure_count += 1
                if drop_failed_rows:
                    dropped_failure_count += 1
                else:
                    existing_rows_to_keep.append(row)

    return ValueResumeState(
        existing_rows_to_keep=existing_rows_to_keep,
        successful_case_keys=successful_case_keys,
        success_count=success_count,
        failure_count=0 if drop_failed_rows else failure_count,
        next_result_case_id=max_case_id + 1,
        dropped_failure_count=dropped_failure_count,
    )


def validate_aspen_paths(
    aspen,
    inputs: Sequence[InputVariable],
    outputs: Sequence[OutputVariable],
    extra_input_path_by_case_column: Optional[Dict[str, str]] = None,
    case_only_input_names: Optional[Sequence[str]] = None,
) -> None:
    missing_items: List[str] = []
    ignored_input_names = set(case_only_input_names or [])

    for item in inputs:
        if item.name in ignored_input_names:
            continue
        if aspen.Tree.FindNode(item.path) is None:
            missing_items.append(f"input '{item.name}': {item.path}")

    for case_column, path in (extra_input_path_by_case_column or {}).items():
        if aspen.Tree.FindNode(path) is None:
            missing_items.append(f"derived input '{case_column}': {path}")

    for item in outputs:
        if aspen.Tree.FindNode(item.path) is None:
            missing_items.append(f"output '{item.name}': {item.path}")

    if missing_items:
        joined = "\n".join(f"- {item}" for item in missing_items)
        raise ValueError(
            "One or more Aspen paths were not found in the loaded model:\n"
            f"{joined}"
        )


def run_aspen_case(
    aspen,
    sleep_after_run_sec: float = 0.0,
    timeout_sec: float = 600.0,
) -> None:
    # Use an asynchronous Aspen run so Python can enforce a per-case timeout.
    try:
        aspen.Engine.Run2(0)
    except TypeError:
        aspen.Engine.Run2()
    except Exception:
        aspen.Engine.Run2()

    wait_for_aspen_to_finish(aspen, timeout_sec=timeout_sec)

    if sleep_after_run_sec > 0:
        time.sleep(sleep_after_run_sec)


def run_case_grid(
    simulation_file: str | Path,
    inputs: Sequence[InputVariable],
    outputs: Sequence[OutputVariable],
    case_grid_csv: Path,
    results_csv: Path,
    total_cases: int,
    visible: bool = False,
    sleep_after_run_sec: float = 0.0,
    extra_input_path_by_case_column: Optional[Dict[str, str]] = None,
    case_only_input_names: Optional[Sequence[str]] = None,
    computed_output_fetchers: Optional[Dict[str, ComputedOutputFetcher]] = None,
    output_value_transformers: Optional[Dict[str, OutputValueTransformer]] = None,
    resume_by_case_values: bool = False,
    drop_failed_existing_results: bool = False,
    case_timeout_sec: float = 600.0,
    force_kill_aspen_on_timeout: bool = False,
) -> RunSummary:
    if not case_grid_csv.exists():
        raise FileNotFoundError(f"Case grid CSV not found: {case_grid_csv}")

    ensure_parent_directory(results_csv)
    if case_timeout_sec <= 0:
        raise ValueError("case_timeout_sec must be greater than 0.")

    ignored_input_names = set(case_only_input_names or [])
    computed_outputs = dict(computed_output_fetchers or {})
    output_transforms = dict(output_value_transformers or {})
    input_path_by_name = {
        item.name: item.path
        for item in inputs
        if item.name not in ignored_input_names
    }
    derived_input_paths = dict(extra_input_path_by_case_column or {})
    for case_column, path in derived_input_paths.items():
        if is_blank(path):
            raise ValueError(
                f"The Aspen path for derived case column '{case_column}' is blank. "
                "Fill in the path in the run script before starting Aspen."
            )

    with case_grid_csv.open("r", newline="", encoding="utf-8-sig") as input_handle:
        reader = csv.DictReader(input_handle)
        case_fieldnames = list(reader.fieldnames or [])
        if not case_fieldnames:
            raise ValueError(f"{case_grid_csv} is missing a header row.")

    output_names = [item.name for item in outputs]
    computed_output_names = list(computed_outputs)
    unknown_output_transforms = sorted(set(output_transforms) - set(output_names))
    if unknown_output_transforms:
        raise ValueError(
            "Output value transformers were provided for names that are not active direct outputs: "
            f"{', '.join(unknown_output_transforms)}"
        )
    duplicate_output_columns = sorted(set(case_fieldnames) & set(output_names))
    if duplicate_output_columns:
        raise ValueError(
            "Case-grid columns and output names must be distinct. "
            f"Conflicting names: {', '.join(duplicate_output_columns)}"
        )
    duplicate_computed_columns = sorted(set(case_fieldnames) & set(computed_output_names))
    if duplicate_computed_columns:
        raise ValueError(
            "Case-grid columns and computed output names must be distinct. "
            f"Conflicting names: {', '.join(duplicate_computed_columns)}"
        )
    duplicate_output_names = sorted(set(output_names) & set(computed_output_names))
    if duplicate_output_names:
        raise ValueError(
            "Direct output names and computed output names must be distinct. "
            f"Conflicting names: {', '.join(duplicate_output_names)}"
        )

    missing_derived_columns = sorted(set(derived_input_paths) - set(case_fieldnames))
    if missing_derived_columns:
        raise ValueError(
            f"The case-grid CSV is missing derived columns required for Aspen input updates: "
            f"{', '.join(missing_derived_columns)}"
        )

    input_setters = dict(input_path_by_name)
    input_setters.update(derived_input_paths)
    fieldnames = [
        *case_fieldnames,
        *output_names,
        *computed_output_names,
        "run_ok",
        "error_message",
        "elapsed_sec",
        "completed_at",
    ]

    resume_state: Optional[ResumeState] = None
    value_resume_state: Optional[ValueResumeState] = None
    successful_case_keys: Set[Tuple[Tuple[str, str], ...]] = set()
    next_result_case_id = 1

    if resume_by_case_values:
        value_resume_state = inspect_existing_results_by_case_values(
            results_csv,
            expected_fieldnames=fieldnames,
            case_fieldnames=case_fieldnames,
            drop_failed_rows=drop_failed_existing_results,
        )
        success_count = value_resume_state.success_count
        failure_count = value_resume_state.failure_count
        successful_case_keys = set(value_resume_state.successful_case_keys)
        next_result_case_id = value_resume_state.next_result_case_id

        pending_case_count = 0
        with case_grid_csv.open("r", newline="", encoding="utf-8-sig") as input_handle:
            reader = csv.DictReader(input_handle)
            for case_row in reader:
                if make_case_value_key(case_row, case_fieldnames) not in successful_case_keys:
                    pending_case_count += 1

        if successful_case_keys:
            print(
                f"Found {len(successful_case_keys)} successful existing case point(s). "
                "Exact matching input points will be skipped.",
                flush=True,
            )
        if value_resume_state.dropped_failure_count > 0:
            print(
                f"Removed {value_resume_state.dropped_failure_count} existing failed/error result row(s) "
                "from the live results CSV.",
                flush=True,
            )
        elif results_csv.exists() and results_csv.stat().st_size > 0:
            print(
                "Existing results CSV did not contain successful matching points, so all current grid points "
                "will be considered for running.",
                flush=True,
            )

        if pending_case_count == 0:
            print("All current grid points already have successful results. Nothing left to run.", flush=True)
            return RunSummary(
                total_cases=total_cases,
                success_count=success_count,
                failure_count=failure_count,
                case_grid_csv=case_grid_csv,
                results_csv=results_csv,
            )
    else:
        resume_state = inspect_existing_results(results_csv, fieldnames, total_cases)
        success_count = resume_state.success_count
        failure_count = resume_state.failure_count

        if resume_state.last_success_case_id > 0:
            print(
                f"Resuming from case {resume_state.next_case_id}/{total_cases} "
                f"using existing results through successful case {resume_state.last_success_case_id}.",
                flush=True,
            )
        elif results_csv.exists() and results_csv.stat().st_size > 0:
            print(
                "Existing results CSV did not contain any successful cases, so the run will restart from case 1.",
                flush=True,
            )

        if resume_state.next_case_id > total_cases:
            print("All cases are already completed successfully. Nothing left to run.", flush=True)
            return RunSummary(
                total_cases=total_cases,
                success_count=success_count,
                failure_count=failure_count,
                case_grid_csv=case_grid_csv,
                results_csv=results_csv,
            )

    aspen = launch_aspen(simulation_file=simulation_file, visible=visible)

    try:
        validate_aspen_paths(
            aspen,
            inputs=inputs,
            outputs=outputs,
            extra_input_path_by_case_column=derived_input_paths,
            case_only_input_names=ignored_input_names,
        )

        with case_grid_csv.open("r", newline="", encoding="utf-8-sig") as input_handle, results_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_handle:
            reader = csv.DictReader(input_handle)
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()

            existing_rows_to_keep = (
                value_resume_state.existing_rows_to_keep
                if resume_by_case_values and value_resume_state is not None
                else resume_state.existing_rows_to_keep
            )
            for existing_row in existing_rows_to_keep:
                writer.writerow(existing_row)
            output_handle.flush()

            for case_row in reader:
                grid_case_id = parse_case_id(
                    case_row.get("case_id", ""),
                    f"Column 'case_id' in {case_grid_csv}",
                )

                if resume_by_case_values:
                    case_value_key = make_case_value_key(case_row, case_fieldnames)
                    if case_value_key in successful_case_keys:
                        print(f"Skipping grid case {grid_case_id}/{total_cases}; point already exists.", flush=True)
                        continue

                    result_case_id = next_result_case_id
                    next_result_case_id += 1
                else:
                    if grid_case_id < resume_state.next_case_id:
                        continue
                    result_case_id = grid_case_id

                started_at = time.time()
                result_row: Dict[str, object] = {}
                for fieldname in case_fieldnames:
                    if fieldname == "case_id":
                        result_row[fieldname] = result_case_id
                    else:
                        result_row[fieldname] = safe_float(case_row[fieldname])

                for output_name in output_names:
                    result_row[output_name] = ""
                for output_name in computed_output_names:
                    result_row[output_name] = ""

                result_row["run_ok"] = 0
                result_row["error_message"] = ""
                result_row["elapsed_sec"] = 0.0
                result_row["completed_at"] = ""
                restart_aspen_after_case = False

                if resume_by_case_values:
                    print(
                        f"Running grid case {grid_case_id}/{total_cases} as result case {result_case_id}...",
                        flush=True,
                    )
                else:
                    print(f"Running case {grid_case_id}/{total_cases}...", flush=True)

                try:
                    for case_column, path in input_setters.items():
                        try:
                            set_node_value(aspen, path, float(case_row[case_column]))
                        except Exception as exc:
                            raise RuntimeError(
                                "Failed while setting Aspen input "
                                f"'{case_column}' at path '{path}' to value {case_row[case_column]!r}: {exc}"
                            ) from exc

                    run_aspen_case(
                        aspen,
                        sleep_after_run_sec=sleep_after_run_sec,
                        timeout_sec=case_timeout_sec,
                    )

                    blank_outputs: List[str] = []
                    for output_item in outputs:
                        raw_value = get_node_value(aspen, output_item.path)
                        transformed_value = raw_value
                        if output_item.name in output_transforms and raw_value is not None:
                            transformed_value = output_transforms[output_item.name](raw_value)
                        result_row[output_item.name] = safe_float(transformed_value)
                        if raw_value is None:
                            blank_outputs.append(output_item.name)

                    for output_name, fetcher in computed_outputs.items():
                        raw_value = fetcher(aspen, dict(result_row))
                        result_row[output_name] = safe_float(raw_value)
                        if raw_value is None:
                            blank_outputs.append(output_name)

                    if blank_outputs:
                        raise RuntimeError(
                            "Aspen returned blank values for the requested outputs after the run: "
                            + ", ".join(blank_outputs)
                        )

                    result_row["run_ok"] = 1
                    success_count += 1
                    if resume_by_case_values:
                        successful_case_keys.add(case_value_key)
                except TimeoutError as exc:
                    result_row["error_message"] = str(exc)
                    failure_count += 1
                    restart_aspen_after_case = True
                    print(
                        "Case timed out. Closing Aspen and starting a fresh Aspen session before continuing...",
                        flush=True,
                    )
                    close_aspen(aspen)
                    if force_kill_aspen_on_timeout:
                        force_kill_aspen_processes()
                    aspen = None
                except Exception as exc:
                    result_row["error_message"] = str(exc)
                    failure_count += 1

                result_row["elapsed_sec"] = round(time.time() - started_at, 3)
                result_row["completed_at"] = datetime.now().isoformat(timespec="seconds")
                writer.writerow(result_row)
                output_handle.flush()

                if restart_aspen_after_case:
                    aspen = launch_aspen(simulation_file=simulation_file, visible=visible)
                    validate_aspen_paths(
                        aspen,
                        inputs=inputs,
                        outputs=outputs,
                        extra_input_path_by_case_column=derived_input_paths,
                        case_only_input_names=ignored_input_names,
                    )

    finally:
        close_aspen(aspen)

    return RunSummary(
        total_cases=total_cases,
        success_count=success_count,
        failure_count=failure_count,
        case_grid_csv=case_grid_csv,
        results_csv=results_csv,
    )
