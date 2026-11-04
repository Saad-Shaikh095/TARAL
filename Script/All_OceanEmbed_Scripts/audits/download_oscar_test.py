"""
OSCAR v2.0 FINAL - SINGLE GRANULE HARMONY TEST
==============================================

Purpose
-------
This test deliberately processes ONLY ONE OSCAR granule.

We first query NASA CMR to find the exact OSCAR FINAL
granule for 2012-01-01.

Then we send that exact granule ID to Harmony.

This removes multi-granule temporal concatenation from
the experiment.

Test:
    Date      : 2012-01-01
    Longitude : 45E -> 105E
    Latitude  : 5N -> 30N
    Variables : u, v, ug, vg

Output:
    D:\\OceanEmbed\\oceanembed-data\\raw\\oscar\\
        oscar_final_v2_single_granule_test.nc

Requirements:
    pip install requests earthaccess
"""

from __future__ import annotations

import sys
from pathlib import Path

import earthaccess
import requests


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\oscar"
)

OUTPUT_FILE = OUTPUT_DIR / (
    "oscar_final_v2_single_granule_test.nc"
)

# OSCAR FINAL v2.0
COLLECTION_ID = "C2098858642-POCLOUD"

SHORT_NAME = "OSCAR_L4_OC_FINAL_V2.0"

# ------------------------------------------------------------
# Test date
# ------------------------------------------------------------

TEST_START = "2012-01-01T00:00:00Z"
TEST_END = "2012-01-01T23:59:59Z"

# ------------------------------------------------------------
# OceanEmbed region
# ------------------------------------------------------------

LON_MIN = 45.0
LON_MAX = 105.0

LAT_MIN = 5.0
LAT_MAX = 30.0

# ------------------------------------------------------------
# Variables
# ------------------------------------------------------------

VARIABLES = [
    "u",
    "v",
    "ug",
    "vg",
]


# ============================================================
# URLS
# ============================================================

CMR_URL = (
    "https://cmr.earthdata.nasa.gov/"
    "search/granules.json"
)

HARMONY_URL = (
    "https://harmony.earthdata.nasa.gov/"
    f"{COLLECTION_ID}/"
    "ogc-api-coverages/1.0.0/"
    "collections/parameter_vars/"
    "coverage/rangeset"
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
                "NASA Earthdata authentication failed."
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
# FIND ONE GRANULE
# ============================================================

def find_granule(session):
    print_header(
        "FINDING OSCAR GRANULE IN CMR"
    )

    print(
        f"Collection : {COLLECTION_ID}"
    )

    print(
        f"Date       : {TEST_START}"
    )

    params = {
        "collection_concept_id": COLLECTION_ID,
        "temporal": (
            f"{TEST_START},{TEST_END}"
        ),
        "page_size": 10,
    }

    print()
    print(
        "Querying NASA CMR..."
    )

    try:

        response = session.get(
            CMR_URL,
            params=params,
            timeout=60,
        )

        print(
            f"CMR HTTP status: "
            f"{response.status_code}"
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:

        print()
        print(
            "CMR request failed."
        )

        print(exc)

        sys.exit(1)

    entries = (
        data
        .get("feed", {})
        .get("entry", [])
    )

    if not entries:

        print()
        print(
            "No OSCAR granule was found."
        )

        print(
            "This would indicate a collection/date "
            "query problem rather than a Harmony problem."
        )

        sys.exit(1)

    print()
    print(
        f"Found {len(entries)} granule(s)."
    )

    # Use the first matching daily granule.
    granule = entries[0]

    granule_id = granule.get(
        "id"
    )

    granule_name = granule.get(
        "producer_granule_id"
    )

    time_start = granule.get(
        "time_start"
    )

    time_end = granule.get(
        "time_end"
    )

    print()
    print(
        "Selected granule:"
    )

    print(
        f"  Granule ID : {granule_id}"
    )

    print(
        f"  Name       : {granule_name}"
    )

    print(
        f"  Start      : {time_start}"
    )

    print(
        f"  End        : {time_end}"
    )

    if not granule_id:

        print()
        print(
            "CMR returned a granule but no "
            "granule ID was found."
        )

        sys.exit(1)

    return granule_id


# ============================================================
# BUILD HARMONY REQUEST
# ============================================================

def build_harmony_params(
    granule_id: str,
) -> list[tuple[str, str]]:

    params = [

        # EXACTLY ONE granule.
        (
            "granuleId",
            granule_id,
        ),

        # Spatial subset.
        (
            "subset",
            f"lat({LAT_MIN}:{LAT_MAX})",
        ),

        (
            "subset",
            f"lon({LON_MIN}:{LON_MAX})",
        ),

        # Variables.
        (
            "variable",
            ",".join(VARIABLES),
        ),

        # NetCDF4.
        (
            "format",
            "application/x-netcdf4",
        ),
    ]

    return params


# ============================================================
# DOWNLOAD ONE GRANULE
# ============================================================

def download_granule(
    session,
    granule_id: str,
):

    print_header(
        "HARMONY SINGLE-GRANULE REQUEST"
    )

    print(
        f"Collection : {COLLECTION_ID}"
    )

    print(
        f"Granule ID : {granule_id}"
    )

    print(
        f"Longitude  : "
        f"{LON_MIN} → {LON_MAX}"
    )

    print(
        f"Latitude   : "
        f"{LAT_MIN} → {LAT_MAX}"
    )

    print(
        f"Variables  : "
        f"{', '.join(VARIABLES)}"
    )

    print(
        "Concatenate: DISABLED"
    )

    params = build_harmony_params(
        granule_id
    )

    # --------------------------------------------------------
    # Print actual URL
    # --------------------------------------------------------

    from requests import Request

    prepared = Request(
        "GET",
        HARMONY_URL,
        params=params,
    ).prepare()

    print()
    print(
        "Harmony URL:"
    )

    print(
        prepared.url
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    print()
    print(
        "Sending request..."
    )

    try:

        response = session.get(
            HARMONY_URL,
            params=params,
            timeout=600,
            allow_redirects=True,
        )

        print(
            f"HTTP status: "
            f"{response.status_code}"
        )

        # ====================================================
        # SUCCESS
        # ====================================================

        if response.ok:

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            temp_file = OUTPUT_FILE.with_suffix(
                ".nc.partial"
            )

            print()
            print(
                "Downloading response..."
            )

            total_bytes = 0

            with open(
                temp_file,
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

            # Atomic rename.
            temp_file.replace(
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

            return True

        # ====================================================
        # ERROR
        # ====================================================

        print()
        print(
            "Harmony returned an error."
        )

        content_type = response.headers.get(
            "content-type",
            "",
        )

        if "application/json" in content_type:

            try:

                error = response.json()

                print()
                print(
                    "Harmony error response:"
                )

                print(
                    error
                )

            except Exception:

                print(
                    response.text[:5000]
                )

        else:

            print(
                response.text[:5000]
            )

        return False

    except requests.RequestException as exc:

        print()
        print_header(
            "HTTP ERROR"
        )

        print(exc)

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "OSCAR v2.0 FINAL"
    )

    print(
        "SINGLE-GRANULE API TEST"
    )

    print()
    print(
        "This test downloads exactly ONE "
        "OSCAR daily granule."
    )

    print(
        "It does NOT use temporal multi-granule "
        "concatenation."
    )

    # --------------------------------------------------------
    # Authenticate
    # --------------------------------------------------------

    authenticate()

    # --------------------------------------------------------
    # Create authenticated HTTP session
    # --------------------------------------------------------

    print_header(
        "CREATING EARTHDATA HTTP SESSION"
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

        print(
            "Could not create authenticated "
            "Earthdata session."
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Find exact granule
    # --------------------------------------------------------

    granule_id = find_granule(
        session
    )

    # --------------------------------------------------------
    # Download through Harmony
    # --------------------------------------------------------

    success = download_granule(
        session,
        granule_id,
    )

    if not success:

        print()
        print_header(
            "TEST FAILED"
        )

        print(
            "The important thing is that we now know "
            "the exact OSCAR granule involved."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print_header(
        "TEST COMPLETE"
    )

    print(
        "The next step is to audit the NetCDF."
    )


if __name__ == "__main__":
    main()