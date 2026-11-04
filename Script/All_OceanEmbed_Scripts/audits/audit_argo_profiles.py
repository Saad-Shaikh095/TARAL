from pathlib import Path
import csv
import json
import multiprocessing as mp
import os
import queue
import time
import traceback


# ============================================================
# CONFIGURATION
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

# Number of files a worker processes before being replaced.
# This helps prevent long-lived native-library state.
FILES_PER_WORKER = 500

# Timeout for one individual NetCDF operation.
FILE_TIMEOUT = 120


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DISCOVER FILES
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
# NETCDF AUDIT
# ============================================================

def audit_one_file(file_path):
    """
    This function runs INSIDE a worker process.

    Important:
    NetCDF/HDF5 operations happen only inside worker
    processes. The parent process never opens a NetCDF file.
    """

    import netCDF4
    import numpy as np

    result = {
        "file": str(file_path),
        "relative_file": str(
            file_path.relative_to(RAW_ROOT)
        ),
        "status": "success",
    }

    try:

        # ----------------------------------------------------
        # Open dataset
        # ----------------------------------------------------

        ds = netCDF4.Dataset(
            str(file_path),
            mode="r"
        )

        try:

            # ------------------------------------------------
            # Basic metadata
            # ------------------------------------------------

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
                list(ds.variables.keys())
            )

            result["global_attributes"] = sorted(
                list(ds.ncattrs())
            )

            # ------------------------------------------------
            # Identify important variables
            # ------------------------------------------------

            important_variables = [
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

            for name in important_variables:

                if name not in ds.variables:
                    continue

                var = ds.variables[name]

                info = {
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
                        list(var.ncattrs())
                    ),
                }

                result[
                    "important_variables"
                ][name] = info

            # ------------------------------------------------
            # Data statistics
            # ------------------------------------------------

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

                    var = ds.variables[name]

                    data = var[:]

                    # Convert masked arrays safely
                    if np.ma.isMaskedArray(data):

                        valid = data.compressed()

                        missing_count = int(
                            np.ma.count_masked(data)
                        )

                    else:

                        data = np.asarray(data)

                        valid = data[
                            np.isfinite(data)
                        ]

                        missing_count = int(
                            data.size - valid.size
                        )

                    stat = {
                        "total_values": int(
                            data.size
                        ),
                        "valid_values": int(
                            valid.size
                        ),
                        "missing_values": missing_count,
                    }

                    if valid.size > 0:

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

                except Exception as e:

                    result[
                        "statistics"
                    ][name] = {
                        "error": str(e)
                    }

            # ------------------------------------------------
            # QC distribution
            # ------------------------------------------------

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

                    data = np.asarray(data)

                    # Convert byte/string QC values safely
                    values = []

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

                        values.append(value)

                    counts = {}

                    for value in values:

                        counts[value] = (
                            counts.get(value, 0) + 1
                        )

                    result["qc"][name] = counts

                except Exception as e:

                    result["qc"][name] = {
                        "error": str(e)
                    }

            # ------------------------------------------------
            # Important metadata values
            # ------------------------------------------------

            metadata_variables = [
                "LATITUDE",
                "LONGITUDE",
                "JULD",
                "PLATFORM_NUMBER",
                "CYCLE_NUMBER",
                "DATA_MODE",
                "DIRECTION",
            ]

            result["metadata_values"] = {}

            for name in metadata_variables:

                if name not in ds.variables:
                    continue

                try:

                    value = ds.variables[name][:]

                    if np.ma.isMaskedArray(value):
                        value = value.filled()

                    # Convert numpy values to JSON-safe form
                    if hasattr(value, "tolist"):
                        value = value.tolist()

                    # Prevent gigantic metadata arrays
                    if isinstance(value, list):

                        if len(value) > 20:
                            value = value[:20]

                    result[
                        "metadata_values"
                    ][name] = value

                except Exception as e:

                    result[
                        "metadata_values"
                    ][name] = {
                        "error": str(e)
                    }

        finally:

            ds.close()

        # ----------------------------------------------------
        # File size
        # ----------------------------------------------------

        result["file_size_bytes"] = (
            file_path.stat().st_size
        )

        return result

    except Exception as e:

        return {
            "file": str(file_path),
            "relative_file": str(
                file_path.relative_to(RAW_ROOT)
            ),
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# WORKER
# ============================================================

def worker_process(
    worker_id,
    task_queue,
    result_queue,
):

    processed = 0

    while processed < FILES_PER_WORKER:

        try:

            file_path = task_queue.get(
                timeout=2
            )

        except queue.Empty:

            break

        if file_path is None:
            break

        # Tell parent which file we're working on
        result_queue.put({
            "type": "started",
            "worker_id": worker_id,
            "file": str(file_path),
        })

        try:

            result = audit_one_file(
                file_path
            )

            result_queue.put({
                "type": "result",
                "worker_id": worker_id,
                "result": result,
            })

        except Exception as e:

            result_queue.put({
                "type": "result",
                "worker_id": worker_id,
                "result": {
                    "file": str(file_path),
                    "relative_file": str(
                        file_path.relative_to(
                            RAW_ROOT
                        )
                    ),
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            })

        processed += 1

    result_queue.put({
        "type": "worker_finished",
        "worker_id": worker_id,
    })


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ARGO NETCDF PROFILE AUDIT V2")
    print("=" * 80)

    print(f"Raw data : {RAW_ROOT}")
    print(f"Output   : {OUTPUT_DIR}")
    print(f"Workers  : {MAX_WORKERS}")
    print(
        f"Worker lifetime : "
        f"{FILES_PER_WORKER} files"
    )

    print()

    files = find_files()

    if not files:

        print("ERROR: No NetCDF files found.")
        return

    # --------------------------------------------------------
    # Load existing checkpoint
    # --------------------------------------------------------

    completed_files = set()

    if RESULTS_FILE.exists():

        print(
            "Loading existing audit checkpoint..."
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
                    pass

        print(
            f"Already audited: "
            f"{len(completed_files):,}"
        )

    # --------------------------------------------------------
    # Determine remaining files
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
            "All files are already audited."
        )

        return

    print()
    print("Starting robust audit...")
    print()

    # --------------------------------------------------------
    # Multiprocessing
    # --------------------------------------------------------

    ctx = mp.get_context("spawn")

    task_queue = ctx.Queue(
        maxsize=MAX_WORKERS * 4
    )

    result_queue = ctx.Queue()

    # --------------------------------------------------------
    # Feed tasks
    # --------------------------------------------------------

    for path in remaining:

        task_queue.put(path)

    for _ in range(MAX_WORKERS):

        task_queue.put(None)

    # --------------------------------------------------------
    # Start workers
    # --------------------------------------------------------

    workers = {}

    for worker_id in range(MAX_WORKERS):

        process = ctx.Process(
            target=worker_process,
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
    # Audit state
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
    # Open checkpoint files
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

    if (
        FAILED_FILE.stat().st_size == 0
    ):

        failed_writer.writerow([
            "file",
            "error_type",
            "error",
        ])

    try:

        while completed < total:

            # ------------------------------------------------
            # Receive worker messages
            # ------------------------------------------------

            try:

                message = result_queue.get(
                    timeout=1
                )

            except queue.Empty:

                message = None

            if message:

                message_type = message["type"]

                # --------------------------------------------
                # Worker started file
                # --------------------------------------------

                if message_type == "started":

                    worker_id = message[
                        "worker_id"
                    ]

                    workers[
                        worker_id
                    ][
                        "current_file"
                    ] = message["file"]

                # --------------------------------------------
                # Worker result
                # --------------------------------------------

                elif message_type == "result":

                    worker_id = message[
                        "worker_id"
                    ]

                    result = message[
                        "result"
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

                    if result["status"] == "success":

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

                # --------------------------------------------
                # Worker finished
                # --------------------------------------------

                elif message_type == "worker_finished":

                    worker_id = message[
                        "worker_id"
                    ]

                    workers[
                        worker_id
                    ][
                        "current_file"
                    ] = None

            # ------------------------------------------------
            # Detect dead workers
            # ------------------------------------------------

            for worker_id, info in list(
                workers.items()
            ):

                process = info[
                    "process"
                ]

                if (
                    not process.is_alive()
                    and process.exitcode != 0
                ):

                    current_file = info[
                        "current_file"
                    ]

                    print()

                    print(
                        f"WARNING: Worker "
                        f"{worker_id} crashed."
                    )

                    print(
                        f"Exit code: "
                        f"{process.exitcode}"
                    )

                    if current_file:

                        print(
                            "The affected file "
                            "will be retried."
                        )

                        task_queue.put(
                            Path(current_file)
                        )

                        # The crashed worker's file
                        # was never checkpointed.

                    # Remove dead worker
                    del workers[
                        worker_id
                    ]

                    # Start replacement worker
                    replacement = ctx.Process(
                        target=worker_process,
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

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            now = time.time()

            if (
                now - last_progress >= 2
            ):

                elapsed = now - start_time

                rate = (
                    completed / elapsed
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

            # ------------------------------------------------
            # Check whether all workers died
            # ------------------------------------------------

            alive = any(
                info["process"].is_alive()
                for info in workers.values()
            )

            if not alive:

                # Give the queue a moment to settle
                time.sleep(0.2)

    finally:

        results_handle.close()
        failed_handle.close()

        # ----------------------------------------------------
        # Shutdown workers
        # ----------------------------------------------------

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
    # FINAL SUMMARY
    # ========================================================

    elapsed = (
        time.time() - start_time
    )

    final_rate = (
        completed / elapsed
        if elapsed > 0
        else 0
    )

    summary = {
        "total_files": total,
        "audited_files": completed,
        "successful": successful,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "files_per_second": final_rate,
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
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"Total files : {total:,}"
    )

    print(
        f"Audited     : {completed:,}"
    )

    print(
        f"Successful  : {successful:,}"
    )

    print(
        f"Errors      : {errors:,}"
    )

    print(
        f"Elapsed     : {elapsed / 60:.2f} minutes"
    )

    print(
        f"Rate        : {final_rate:.2f} files/s"
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
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()