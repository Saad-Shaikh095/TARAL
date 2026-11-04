"""
OSCAR v2.0 FINAL
Native Single-Granule Download Test
====================================

Downloads exactly ONE OSCAR FINAL v2.0 native granule
using the authenticated NASA Earthdata HTTP session.

Test granule:
    oscar_currents_final_20120101.nc

Purpose:
    Verify direct PO.DAAC protected-archive access and
    inspect the real native OSCAR NetCDF before designing
    the production 2012-2024 collector.

No Harmony.
No concatenation.
No spatial subsetting.

Requirements:
    pip install earthaccess requests
"""

from __future__ import annotations

import sys
from pathlib import Path

import earthaccess


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\oscar\test"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "oscar_currents_final_20120101.nc"
)

DOWNLOAD_URL = (
    "https://archive.podaac.earthdata.nasa.gov/"
    "podaac-ops-cumulus-protected/"
    "OSCAR_L4_OC_FINAL_V2.0/"
    "oscar_currents_final_20120101.nc"
)


# ============================================================
# LOGGING
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# AUTHENTICATION
# ============================================================

def authenticate():

    print_header(
        "NASA EARTHDATA AUTHENTICATION"
    )

    print(
        "Authenticating with NASA Earthdata..."
    )

    try:

        auth = earthaccess.login(
            persist=True,
            strategy="interactive",
        )

        if not auth.authenticated:

            raise RuntimeError(
                "Earthdata authentication failed."
            )

        print(
            "Authentication successful."
        )

        return auth

    except Exception as exc:

        print()
        print(
            "Authentication failed."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)


# ============================================================
# DOWNLOAD
# ============================================================

def download():

    print_header(
        "OSCAR NATIVE GRANULE DOWNLOAD"
    )

    print(
        "Dataset    : OSCAR_L4_OC_FINAL_V2.0"
    )

    print(
        "Granule    : oscar_currents_final_20120101.nc"
    )

    print(
        "Date       : 2012-01-01"
    )

    print(
        f"Output     : {OUTPUT_FILE}"
    )

    print()
    print(
        "Download URL:"
    )

    print(
        DOWNLOAD_URL
    )

    # --------------------------------------------------------
    # Create authenticated session
    # --------------------------------------------------------

    print()
    print(
        "Creating authenticated Earthdata HTTP session..."
    )

    try:

        session = (
            earthaccess
            .get_requests_https_session()
        )

        print(
            "Authenticated session ready."
        )

    except Exception as exc:

        print()
        print(
            "Could not create Earthdata HTTP session."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = OUTPUT_FILE.with_suffix(
        ".nc.partial"
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    print()
    print(
        "Sending download request..."
    )

    try:

        with session.get(
            DOWNLOAD_URL,
            stream=True,
            timeout=300,
        ) as response:

            print(
                f"HTTP status: "
                f"{response.status_code}"
            )

            response.raise_for_status()

            # -----------------------------------------------
            # Server metadata
            # -----------------------------------------------

            content_length = response.headers.get(
                "Content-Length"
            )

            content_type = response.headers.get(
                "Content-Type"
            )

            print(
                f"Content-Type: "
                f"{content_type}"
            )

            if content_length:

                expected_mb = (
                    int(content_length)
                    / (1024 * 1024)
                )

                print(
                    f"Expected size: "
                    f"{expected_mb:,.2f} MB"
                )

            # -----------------------------------------------
            # Stream to disk
            # -----------------------------------------------

            print()
            print(
                "Downloading..."
            )

            total_bytes = 0

            with open(
                temporary_file,
                "wb",
            ) as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue

                    file.write(chunk)

                    total_bytes += len(
                        chunk
                    )

                    size_mb = (
                        total_bytes
                        / (1024 * 1024)
                    )

                    print(
                        f"\rDownloaded: "
                        f"{size_mb:,.2f} MB",
                        end="",
                        flush=True,
                    )

            print()

        # ----------------------------------------------------
        # Atomic rename
        # ----------------------------------------------------

        temporary_file.replace(
            OUTPUT_FILE
        )

        final_size_mb = (
            OUTPUT_FILE.stat().st_size
            / (1024 * 1024)
        )

        print()
        print_header(
            "DOWNLOAD SUCCESSFUL"
        )

        print(
            f"File size : "
            f"{final_size_mb:,.2f} MB"
        )

        print(
            f"File      : "
            f"{OUTPUT_FILE}"
        )

    except Exception as exc:

        print()

        # Remove incomplete file.
        if temporary_file.exists():

            try:
                temporary_file.unlink()
            except OSError:
                pass

        print_header(
            "DOWNLOAD FAILED"
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "OSCAR v2.0 FINAL"
    )

    print(
        "NATIVE SINGLE-GRANULE TEST"
    )

    print()
    print(
        "This downloads exactly one real OSCAR file."
    )

    print(
        "No full historical collection will be downloaded."
    )

    authenticate()

    download()

    print()
    print_header(
        "TEST COMPLETE"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Audit the downloaded NetCDF file."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()