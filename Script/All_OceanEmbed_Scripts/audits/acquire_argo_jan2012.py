from pathlib import Path
import csv
import json
import multiprocessing as mp
import queue
import time
import traceback


# ============================================================
# CONFIG
# ============================================================

RAW_ROOT = Path(
    r"D:\OceanEmbed\argo\raw\2012_01"
)

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\argo\processed\audit_2012_01"
)

RESULTS_FILE = OUTPUT_DIR / "profile_audit.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "profile_audit_summary.json"
FAILED_FILE = OUTPUT_DIR / "profile_audit_failed.csv"

MAX_WORKERS = 4

# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND FILES
# ============================================================

def find_files():

    print("Finding NetCDF files...")

    files = sorted(
        RAW_ROOT.rglob("*.nc")
    )

    print(
        f"NetCDF files found: {len(files):,}"
    )

    return files


# ============================================================
# AUDIT ONE FILE
# ============================================================

def audit_one_file(file_path):

    import netCDF4
    import numpy as np

    result = {
        "file": str(file_path),
        "relative_file": str(
            file_path.relative_to(RAW_ROOT)
        ),
        "status": "success",
    }

    ds = None

    try:

        # ----------------------------------------------------
        # Open NetCDF
        # ----------------------------------------------------

        ds = netCDF4.Dataset(
            str(file_path),
            mode="r"
        )

        # ----------------------------------------------------
        # Basic information
        # ----------------------------------------------------

        result["format"] = getattr(
            ds,
            "data_model",
            None
        )

        result["dimensions"] = {
            name: len(dim)
            for name, dim in ds.dimensions.items()
        }

        result["variables"] = sorted(
            ds.variables.keys()
        )

        result["global_attributes"] = sorted(
            ds.ncattrs()
        )

        # ----------------------------------------------------
        # Important variables
        # ----------------------------------------------------

        important = [
            "PRES",
            "PRES_QC",
            "PRES_ADJUSTED",
            "PRES_ADJUSTED_QC",

            "TEMP",
            "TEMP_QC",
            "TEMP_ADJUSTED",
            "TEMP_ADJUSTED_QC",

            "PSAL",
            "PSAL_QC",
            "PSAL_ADJUSTED",
            "PSAL_ADJUSTED_QC",

            "JULD",
            "JULD_QC",

            "LATITUDE",
            "LONGITUDE",

            "PLATFORM_NUMBER",
            "CYCLE_NUMBER",
            "DATA_MODE",
            "DIRECTION",
        ]

        result["important_variables"] = {}

        for name in important:

            if name not in ds.variables:
                continue

            var = ds.variables[name]

            result[
                "important_variables"
            ][name] = {
                "dimensions": list(
                    var.dimensions
                ),
                "shape": list(
                    var.shape
                ),
                "dtype": str(
                    var.dtype
                ),
                "attributes": sorted(
                    var.ncattrs()
                ),
            }

        # ----------------------------------------------------
        # Numeric statistics
        # ----------------------------------------------------

        result["statistics"] = {}

        for name in [
            "PRES",
            "TEMP",
            "PSAL",
            "PRES_ADJUSTED",
            "TEMP_ADJUSTED",
            "PSAL_ADJUSTED",
        ]:

            if name not in ds.variables:
                continue

            try:

                data = ds.variables[name][:]

                if np.ma.isMaskedArray(data):

                    valid = data.compressed()

                    missing = int(
                        np.ma.count_masked(data)
                    )

                else:

                    data = np.asarray(data)

                    valid = data[
                        np.isfinite(data)
                    ]

                    missing = int(
                        data.size - valid.size
                    )

                stat = {
                    "total_values": int(
                        data.size
                    ),
                    "valid_values": int(
                        valid.size
                    ),
                    "missing_values": missing,
                }

                if valid.size:

                    stat["min"] = float(
                        np.min(valid)
                    )

                    stat["max"] = float(
                        np.max(valid)
                    )

                    stat["mean"] = float(
                        np.mean(valid)
                    )

                result[
                    "statistics"
                ][name] = stat

            except Exception as exc:

                result[
                    "statistics"
                ][name] = {
                    "error": str(exc)
                }

        # ----------------------------------------------------
        # QC distributions
        # ----------------------------------------------------

        result["qc"] = {}

        for name in [
            "PRES_QC",
            "TEMP_QC",
            "PSAL_QC",
            "PRES_ADJUSTED_QC",
            "TEMP_ADJUSTED_QC",
            "PSAL_ADJUSTED_QC",
        ]:

            if name not in ds.variables:
                continue

            try:

                data = ds.variables[name][:]

                if np.ma.isMaskedArray(data):
                    data = data.compressed()

                counts = {}

                for value in data.flatten():

                    if isinstance(
                        value,
                        bytes
                    ):
                        value = value.decode(
                            "ascii",
                            errors="replace"
                        )
                    else:
                        value = str(value)

                    counts[value] = (
                        counts.get(value, 0) + 1
                    )

                result["qc"][name] = counts

            except Exception as exc:

                result["qc"][name] = {
                    "error": str(exc)
                }

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        result["metadata_values"] = {}

        for name in [
            "LATITUDE",
            "LONGITUDE",
            "JULD",
            "PLATFORM_NUMBER",
            "CYCLE_NUMBER",
            "DATA_MODE",
            "DIRECTION",
        ]:

            if name not in ds.variables:
                continue

            try:

                value = ds.variables[name][:]

                if np.ma.isMaskedArray(value):
                    value = value.filled()

                if hasattr(value, "tolist"):
                    value = value.tolist()

                # Keep metadata compact
                if isinstance(value, list):
                    value = value[:20]

                result[
                    "metadata_values"
                ][name] = value

            except Exception as exc:

                result[
                    "metadata_values"
                ][name] = {
                    "error": str(exc)
                }

        result["file_size_bytes"] = (
            file_path.stat().st_size
        )

        return result

    except Exception as exc:

        return {
            "file": str(file_path),
            "relative_file": str(
                file_path.relative_to(RAW_ROOT)
            ),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    finally:

        if ds is not None:

            try:
                ds.close()
            except Exception:
                pass


# ============================================================
# WORKER PROCESS
# ============================================================

def worker(
    worker_id,
    task_queue,
    result_queue
):

    while True:

        try:

            file_path = task_queue.get()

        except (EOFError, OSError):

            break

        # Sentinel
        if file_path is None:

            break

        try:

            result = audit_one_file(
                file_path
            )

            result_queue.put({
                "type": "result",
                "worker_id": worker_id,
                "result": result,
            })

        except BaseException as exc:

            # This handles Python-level exceptions.
            # A true native crash cannot be caught here,
            # but the parent process will detect the worker
            # disappearance.
            result_queue.put({
                "type": "worker_error",
                "worker_id": worker_id,
                "file": str(file_path),
                "error": str(exc),
            })


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ARGO NETCDF PROFILE AUDIT V3")
    print("=" * 80)

    print(
        f"Raw data : {RAW_ROOT}"
    )

    print(
        f"Output   : {OUTPUT_DIR}"
    )

    print(
        f"Workers  : {MAX_WORKERS}"
    )

    print()

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    files = find_files()

    if not files:

        print(
            "ERROR: No NetCDF files found."
        )

        return

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    completed_files = set()

    if RESULTS_FILE.exists():

        print(
            "Loading existing checkpoint..."
        )

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                try:

                    record = json.loads(
                        line
                    )

                    completed_files.add(
                        record["file"]
                    )

                except Exception:
                    continue

        print(
            f"Already audited: "
            f"{len(completed_files):,}"
        )

    # --------------------------------------------------------
    # Remaining
    # --------------------------------------------------------

    remaining = [
        path
        for path in files
        if str(path) not in completed_files
    ]

    print(
        f"Remaining: {len(remaining):,}"
    )

    if not remaining:

        print()
        print(
            "Everything is already audited."
        )

        return

    print()
    print(
        "Starting robust audit..."
    )
    print()

    # --------------------------------------------------------
    # Multiprocessing context
    # --------------------------------------------------------

    ctx = mp.get_context("spawn")

    task_queue = ctx.Queue()

    result_queue = ctx.Queue()

    # --------------------------------------------------------
    # Feed tasks
    #
    # IMPORTANT:
    # Queue is unbounded.
    # Workers are started BEFORE we feed the queue.
    # --------------------------------------------------------

    workers = {}

    for worker_id in range(MAX_WORKERS):

        process = ctx.Process(
            target=worker,
            args=(
                worker_id,
                task_queue,
                result_queue,
            ),
        )

        process.start()

        workers[worker_id] = {
            "process": process,
            "current_file": None,
        }

    # --------------------------------------------------------
    # Put work AFTER workers are alive
    # --------------------------------------------------------

    for path in remaining:

        task_queue.put(path)

    # Stop signals
    for _ in range(MAX_WORKERS):

        task_queue.put(None)

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    total = len(files)

    completed = len(
        completed_files
    )

    successful = 0
    errors = 0

    start_time = time.time()
    last_progress = start_time

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    results_handle = open(
        RESULTS_FILE,
        "a",
        encoding="utf-8"
    )

    failed_handle = open(
        FAILED_FILE,
        "a",
        encoding="utf-8"
    )

    failed_writer = csv.writer(
        failed_handle
    )

    if FAILED_FILE.stat().st_size == 0:

        failed_writer.writerow([
            "file",
            "error_type",
            "error",
        ])

    try:

        while completed < total:

            # ----------------------------------------------
            # Receive result
            # ----------------------------------------------

            try:

                message = result_queue.get(
                    timeout=1
                )

            except queue.Empty:

                message = None

            if message:

                message_type = message[
                    "type"
                ]

                # ------------------------------------------
                # Normal result
                # ------------------------------------------

                if message_type == "result":

                    result = message[
                        "result"
                    ]

                    worker_id = message[
                        "worker_id"
                    ]

                    file_path = result[
                        "file"
                    ]

                    # Checkpoint immediately
                    results_handle.write(
                        json.dumps(
                            result,
                            default=str
                        )
                        + "\n"
                    )

                    results_handle.flush()

                    completed += 1

                    workers[
                        worker_id
                    ][
                        "current_file"
                    ] = None

                    if result[
                        "status"
                    ] == "success":

                        successful += 1

                    else:

                        errors += 1

                        failed_writer.writerow([
                            file_path,
                            result.get(
                                "error_type",
                                ""
                            ),
                            result.get(
                                "error",
                                ""
                            ),
                        ])

                        failed_handle.flush()

                # ------------------------------------------
                # Python-level worker error
                # ------------------------------------------

                elif message_type == "worker_error":

                    errors += 1

            # ----------------------------------------------
            # Detect dead workers
            # ----------------------------------------------

            for worker_id in list(
                workers.keys()
            ):

                info = workers[
                    worker_id
                ]

                process = info[
                    "process"
                ]

                if (
                    not process.is_alive()
                    and process.exitcode != 0
                ):

                    print()

                    print(
                        f"WARNING: Worker "
                        f"{worker_id} crashed."
                    )

                    print(
                        f"Exit code: "
                        f"{process.exitcode}"
                    )

                    # --------------------------------------
                    # Restart worker
                    # --------------------------------------

                    replacement = ctx.Process(
                        target=worker,
                        args=(
                            worker_id,
                            task_queue,
                            result_queue,
                        ),
                    )

                    replacement.start()

                    workers[
                        worker_id
                    ] = {
                        "process": replacement,
                        "current_file": None,
                    }

                    print(
                        f"Worker "
                        f"{worker_id} restarted."
                    )

            # ----------------------------------------------
            # Progress
            # ----------------------------------------------

            now = time.time()

            if (
                now - last_progress >= 2
            ):

                elapsed = (
                    now - start_time
                )

                rate = (
                    (completed - len(
                        completed_files
                    )) / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"\rAudited: "
                    f"{completed:,}/{total:,} "
                    f"| Success: {successful:,} "
                    f"| Errors: {errors:,} "
                    f"| Rate: {rate:.1f} files/s",
                    end="",
                    flush=True
                )

                last_progress = now

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Audit interrupted by user."
        )

    finally:

        results_handle.close()
        failed_handle.close()

        # ----------------------------------------------
        # Stop workers
        # ----------------------------------------------

        for info in workers.values():

            process = info[
                "process"
            ]

            if process.is_alive():

                process.terminate()

            process.join(
                timeout=5
            )

    # ========================================================
    # FINAL
    # ========================================================

    elapsed = (
        time.time() - start_time
    )

    processed_this_run = (
        completed - len(completed_files)
    )

    rate = (
        processed_this_run / elapsed
        if elapsed > 0
        else 0
    )

    summary = {
        "total_files": total,
        "completed_files": completed,
        "successful": successful,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "files_per_second": rate,
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print()
    print("=" * 80)
    print("AUDIT RUN FINISHED")
    print("=" * 80)

    print(
        f"Total files : {total:,}"
    )

    print(
        f"Completed   : {completed:,}"
    )

    print(
        f"Successful  : {successful:,}"
    )

    print(
        f"Errors      : {errors:,}"
    )

    print(
        f"Time        : {elapsed / 60:.2f} minutes"
    )

    print(
        f"Rate        : {rate:.2f} files/s"
    )

    print()

    print(
        f"Results : {RESULTS_FILE}"
    )

    print(
        f"Failures: {FAILED_FILE}"
    )

    print(
        f"Summary : {SUMMARY_FILE}"
    )

    print("=" * 80)


# ============================================================
# WINDOWS
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()