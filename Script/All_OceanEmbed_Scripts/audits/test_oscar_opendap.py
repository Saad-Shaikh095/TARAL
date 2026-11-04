from pathlib import Path
from getpass import getpass
from urllib.parse import quote
import os
import sys
import time

import earthaccess
import numpy as np
import xarray as xr


# ============================================================
# OSCAR OPeNDAP SUBSETTING VERIFICATION
# ============================================================

GRANULE = "oscar_currents_final_20130224"

OPENDAP_BASE = (
    "https://opendap.earthdata.nasa.gov/"
    "collections/C2098858642-POCLOUD/"
    f"granules/{GRANULE}"
)

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\oscar\test"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    f"{GRANULE}_opendap_uv_subset.nc"
)

NATIVE_FILE = Path(
    r"D:\OceanEmbed\oceanembed-data\data\raw\test"
) / f"{GRANULE}.nc"


# ============================================================
# REQUESTED OSCAR NATIVE REGION
# ============================================================
#
# OSCAR native grid:
#
# Longitude:
#   0.00, 0.25, 0.50, ..., 359.75
#
# Latitude:
#   -89.75, -89.50, ..., 89.75
#
# Variable dimension order:
#   u(time, longitude, latitude)
#   v(time, longitude, latitude)
#
# Requested region:
#   Longitude: 45.00E -> 105.00E
#   Latitude : 5.25N  -> 29.75N
#
# Longitude:
#   45.00 / 0.25 = 180
#   105.00 / 0.25 = 420
#
# Latitude:
#   -89.75 + index * 0.25
#
#   5.25  -> index 380
#   29.75 -> index 478
#
# Therefore:
#   longitude = 241 points
#   latitude  = 99 points
#   time      = 1
#
# IMPORTANT:
#   Python/xarray slicing below uses END + 1 because the
#   native extraction is inclusive.
#
# ============================================================

LON_START = 180
LON_END = 420

LAT_START = 380
LAT_END = 478

EXPECTED_TIME = 1
EXPECTED_LON = LON_END - LON_START + 1
EXPECTED_LAT = LAT_END - LAT_START + 1

# Coordinate variables in the native OSCAR file are float64.
# A very small tolerance is still allowed for representation.
COORD_ATOL = 1e-10
COORD_RTOL = 0.0

# Current values should match exactly in this verification.
VALUE_ATOL = 1e-12
VALUE_RTOL = 1e-10


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def authenticate_earthdata():
    print_header("[1/6] EARTHDATA AUTHENTICATION")

    print()
    print("Enter your NASA Earthdata Login credentials.")
    print()

    username = input("Earthdata username:\n> ").strip()

    if not username:
        raise RuntimeError(
            "Earthdata username cannot be empty."
        )

    print()

    password = getpass(
        "Earthdata password:\n> "
    )

    if not password:
        raise RuntimeError(
            "Earthdata password cannot be empty."
        )

    # Use the credentials supplied above through the
    # environment strategy so earthaccess does not prompt
    # a second time.
    os.environ["EARTHDATA_USERNAME"] = username
    os.environ["EARTHDATA_PASSWORD"] = password

    print()
    print("Authenticating with Earthdata...")

    auth = earthaccess.login(
        strategy="environment",
        persist=False,
    )

    if not auth:
        raise RuntimeError(
            "Earthdata authentication failed."
        )

    print("Authentication OK.")

    return auth


def validate_coordinate_structure(
    lon: np.ndarray,
    lat: np.ndarray,
) -> None:

    if lon.ndim != 1:
        raise RuntimeError(
            f"Longitude must be 1D, got {lon.ndim}D."
        )

    if lat.ndim != 1:
        raise RuntimeError(
            f"Latitude must be 1D, got {lat.ndim}D."
        )

    if len(lon) != EXPECTED_LON:
        raise RuntimeError(
            f"Expected {EXPECTED_LON} longitude points, "
            f"got {len(lon)}."
        )

    if len(lat) != EXPECTED_LAT:
        raise RuntimeError(
            f"Expected {EXPECTED_LAT} latitude points, "
            f"got {len(lat)}."
        )

    if not np.allclose(
        np.diff(lon),
        0.25,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            "Longitude spacing is not 0.25 degrees."
        )

    if not np.allclose(
        np.diff(lat),
        0.25,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            "Latitude spacing is not 0.25 degrees."
        )

    if not np.isclose(
        lon[0],
        45.0,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            f"Unexpected first longitude: {lon[0]}"
        )

    if not np.isclose(
        lon[-1],
        105.0,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            f"Unexpected last longitude: {lon[-1]}"
        )

    if not np.isclose(
        lat[0],
        5.25,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            f"Unexpected first latitude: {lat[0]}"
        )

    if not np.isclose(
        lat[-1],
        29.75,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            f"Unexpected last latitude: {lat[-1]}"
        )


def compare_coordinates(
    native_lon: np.ndarray,
    native_lat: np.ndarray,
    subset_lon: np.ndarray,
    subset_lat: np.ndarray,
):
    lon_diff = np.abs(
        native_lon - subset_lon
    )

    lat_diff = np.abs(
        native_lat - subset_lat
    )

    lon_match = np.allclose(
        native_lon,
        subset_lon,
        atol=COORD_ATOL,
        rtol=COORD_RTOL,
    )

    lat_match = np.allclose(
        native_lat,
        subset_lat,
        atol=COORD_ATOL,
        rtol=COORD_RTOL,
    )

    print()
    print("Coordinate comparison:")

    print()
    print("Longitude:")
    print(f"  match       : {lon_match}")
    print(
        f"  max diff    : {np.max(lon_diff):.17g}"
    )
    print(
        f"  mean diff   : {np.mean(lon_diff):.17g}"
    )

    print()
    print("Latitude:")
    print(f"  match       : {lat_match}")
    print(
        f"  max diff    : {np.max(lat_diff):.17g}"
    )
    print(
        f"  mean diff   : {np.mean(lat_diff):.17g}"
    )

    return lon_match, lat_match


def compare_data(
    native_data: np.ndarray,
    subset_data: np.ndarray,
    name: str,
):
    if native_data.shape != subset_data.shape:
        print(
            f"{name} shape match : False"
        )
        return False

    # xarray should decode OSCAR's -999 fill values
    # to NaN. Treat finite/non-finite structure as
    # part of the equivalence test.
    native_finite = np.isfinite(native_data)
    subset_finite = np.isfinite(subset_data)

    mask_match = np.array_equal(
        native_finite,
        subset_finite,
    )

    valid_count = int(
        np.count_nonzero(native_finite)
    )

    native_missing = int(
        np.count_nonzero(~native_finite)
    )

    subset_missing = int(
        np.count_nonzero(~subset_finite)
    )

    print()
    print(f"{name} data comparison:")
    print(
        f"  shape match : "
        f"{native_data.shape == subset_data.shape}"
    )
    print(
        f"  mask match  : {mask_match}"
    )
    print(
        f"  native valid: {valid_count:,}"
    )
    print(
        f"  subset valid: "
        f"{np.count_nonzero(subset_finite):,}"
    )
    print(
        f"  native miss : {native_missing:,}"
    )
    print(
        f"  subset miss : {subset_missing:,}"
    )

    if not mask_match:
        print(
            "  numerical match: False"
        )
        return False

    # Compare only cells that are valid in both datasets.
    valid_both = (
        native_finite &
        subset_finite
    )

    if np.any(valid_both):

        differences = np.abs(
            native_data[valid_both]
            - subset_data[valid_both]
        )

        max_abs_diff = float(
            np.max(differences)
        )

        mean_abs_diff = float(
            np.mean(differences)
        )

        numerical_match = np.allclose(
            native_data[valid_both],
            subset_data[valid_both],
            atol=VALUE_ATOL,
            rtol=VALUE_RTOL,
        )

    else:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
        numerical_match = True

    print(
        f"  numerical match: "
        f"{numerical_match}"
    )

    print(
        f"  max abs diff   : "
        f"{max_abs_diff:.15g}"
    )

    print(
        f"  mean abs diff  : "
        f"{mean_abs_diff:.15g}"
    )

    return (
        mask_match and
        numerical_match
    )


def format_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "OSCAR OPeNDAP SUBSETTING VERIFICATION"
    )

    print()
    print("Granule:")
    print(f"  {GRANULE}")

    print()
    print("OPeNDAP endpoint:")
    print(f"  {OPENDAP_BASE}")

    print()
    print("Requested variables:")
    print("  u")
    print("  v")

    print()
    print("Requested native region:")
    print("  Longitude: 45.00E to 105.00E")
    print("  Latitude : 5.25N to 29.75N")

    print()
    print("Expected native subset:")
    print(f"  longitude = {EXPECTED_LON}")
    print(f"  latitude  = {EXPECTED_LAT}")
    print(f"  time      = {EXPECTED_TIME}")

    print()
    print("Native comparison file:")
    print(f"  {NATIVE_FILE}")

    if not NATIVE_FILE.exists():
        raise FileNotFoundError(
            "Native OSCAR comparison file does not exist:\n"
            f"{NATIVE_FILE}"
        )

    native_size = NATIVE_FILE.stat().st_size

    print()
    print(
        f"Native file size: "
        f"{native_size:,} bytes"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Authentication
    # --------------------------------------------------------

    authenticate_earthdata()

    # --------------------------------------------------------
    # 2. Authenticated HTTP session
    # --------------------------------------------------------

    print_header(
        "[2/6] CREATING AUTHENTICATED HTTP SESSION"
    )

    session = (
        earthaccess.get_requests_https_session()
    )

    if session is None:
        raise RuntimeError(
            "Could not create authenticated "
            "Earthdata HTTP session."
        )

    print(
        "Authenticated HTTP session created."
    )

    # --------------------------------------------------------
    # 3. Server-side OPeNDAP subset
    # --------------------------------------------------------

    print_header(
        "[3/6] REQUESTING SERVER-SIDE SUBSET"
    )

    constraint = (
        f"/u[0][{LON_START}:{LON_END}]"
        f"[{LAT_START}:{LAT_END}];"
        f"/v[0][{LON_START}:{LON_END}]"
        f"[{LAT_START}:{LAT_END}];"
        f"/lon[{LON_START}:{LON_END}];"
        f"/lat[{LAT_START}:{LAT_END}]"
    )

    print()
    print("Constraint:")
    print(f"  {constraint}")

    print()
    print("Expected result:")
    print(
        f"  u: "
        f"(1, {EXPECTED_LON}, {EXPECTED_LAT})"
    )
    print(
        f"  v: "
        f"(1, {EXPECTED_LON}, {EXPECTED_LAT})"
    )

    encoded_constraint = quote(
        constraint,
        safe="",
    )

    request_url = (
        OPENDAP_BASE +
        ".dap.nc4?dap4.ce=" +
        encoded_constraint
    )

    print()
    print(
        "Sending request to NASA OPeNDAP..."
    )

    request_start = time.perf_counter()

    response = session.get(
        request_url,
        stream=True,
        timeout=(30, 300),
    )

    header_time = (
        time.perf_counter() -
        request_start
    )

    print()
    print("HTTP response:")
    print(
        f"  Status         : "
        f"{response.status_code}"
    )
    print(
        f"  Content-Type   : "
        f"{response.headers.get('Content-Type')}"
    )
    print(
        f"  Content-Length : "
        f"{response.headers.get('Content-Length')}"
    )
    print(
        f"  Header time    : "
        f"{header_time:.2f} seconds"
    )

    response.raise_for_status()

    print()
    print(
        "Downloading returned subset..."
    )

    download_start = time.perf_counter()

    total_bytes = 0

    with open(
        OUTPUT_FILE,
        "wb",
    ) as output_handle:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if not chunk:
                continue

            output_handle.write(chunk)
            total_bytes += len(chunk)

    download_time = (
        time.perf_counter() -
        download_start
    )

    total_time = (
        time.perf_counter() -
        request_start
    )

    print()
    print("Download complete.")

    print(f"  Output : {OUTPUT_FILE}")
    print(f"  Bytes  : {total_bytes:,}")
    print(
        f"  MiB    : "
        f"{format_mib(total_bytes):.3f}"
    )
    print(
        f"  Time   : "
        f"{total_time:.2f} seconds"
    )

    if total_bytes == 0:
        raise RuntimeError(
            "OPeNDAP returned an empty file."
        )

    # --------------------------------------------------------
    # 4. Open returned NetCDF
    # --------------------------------------------------------

    print_header(
        "[4/6] OPENING RETURNED NETCDF"
    )

    ds = xr.open_dataset(
        OUTPUT_FILE,
        decode_cf=True,
        mask_and_scale=True,
    )

    try:

        print()
        print("Dataset:")
        print(ds)

        # ----------------------------------------------------
        # Required variables
        # ----------------------------------------------------

        required_variables = {
            "u",
            "v",
            "lon",
            "lat",
        }

        missing_variables = (
            required_variables -
            set(ds.variables)
        )

        if missing_variables:
            raise RuntimeError(
                "Returned dataset is missing "
                f"variables: {sorted(missing_variables)}"
            )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        print_header(
            "[5/6] VALIDATING SUBSET STRUCTURE"
        )

        time_size = ds.sizes.get(
            "time",
            0,
        )

        lon_size = ds.sizes.get(
            "longitude",
            0,
        )

        lat_size = ds.sizes.get(
            "latitude",
            0,
        )

        print()
        print("Dimensions:")
        print(f"  time      : {time_size}")
        print(f"  longitude : {lon_size}")
        print(f"  latitude  : {lat_size}")

        if time_size != EXPECTED_TIME:
            raise RuntimeError(
                f"Expected time={EXPECTED_TIME}, "
                f"got {time_size}"
            )

        if lon_size != EXPECTED_LON:
            raise RuntimeError(
                f"Expected longitude={EXPECTED_LON}, "
                f"got {lon_size}"
            )

        if lat_size != EXPECTED_LAT:
            raise RuntimeError(
                f"Expected latitude={EXPECTED_LAT}, "
                f"got {lat_size}"
            )

        # ----------------------------------------------------
        # Variable dimensions
        # ----------------------------------------------------

        print()
        print("u dimensions:")
        print(f"  {ds['u'].dims}")

        print()
        print("v dimensions:")
        print(f"  {ds['v'].dims}")

        expected_dims = (
            "time",
            "longitude",
            "latitude",
        )

        if ds["u"].dims != expected_dims:
            raise RuntimeError(
                f"Unexpected u dimensions: "
                f"{ds['u'].dims}"
            )

        if ds["v"].dims != expected_dims:
            raise RuntimeError(
                f"Unexpected v dimensions: "
                f"{ds['v'].dims}"
            )

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        lon = np.asarray(
            ds["lon"].values
        )

        lat = np.asarray(
            ds["lat"].values
        )

        print()
        print("Longitude:")
        print(f"  count : {len(lon)}")
        print(f"  first : {lon[0]:.17g}")
        print(f"  last  : {lon[-1]:.17g}")

        print()
        print("Latitude:")
        print(f"  count : {len(lat)}")
        print(f"  first : {lat[0]:.17g}")
        print(f"  last  : {lat[-1]:.17g}")

        validate_coordinate_structure(
            lon,
            lat,
        )

        print()
        print(
            "Coordinate structure: PASS"
        )

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        subset_u = np.asarray(
            ds["u"].isel(time=0).values
        )

        subset_v = np.asarray(
            ds["v"].isel(time=0).values
        )

        def print_stats(
            name: str,
            data: np.ndarray,
        ):

            finite = np.isfinite(data)

            valid = int(
                np.count_nonzero(finite)
            )

            missing = int(
                np.count_nonzero(~finite)
            )

            print()
            print(f"{name} statistics:")
            print(f"  shape   : {data.shape}")
            print(f"  dtype   : {data.dtype}")
            print(f"  valid   : {valid:,}")
            print(f"  missing : {missing:,}")

            if valid:
                print(
                    f"  min     : "
                    f"{np.nanmin(data):.8f}"
                )
                print(
                    f"  max     : "
                    f"{np.nanmax(data):.8f}"
                )
                print(
                    f"  mean    : "
                    f"{np.nanmean(data):.8f}"
                )

        print_stats(
            "u",
            subset_u,
        )

        print_stats(
            "v",
            subset_v,
        )

        # ----------------------------------------------------
        # 6. Native comparison
        # ----------------------------------------------------

        print_header(
            "[6/6] COMPARING AGAINST EXACT NATIVE FILE"
        )

        print()
        print("Native file:")
        print(f"  {NATIVE_FILE}")

        print()
        print("Opening native OSCAR file...")

        native = xr.open_dataset(
            NATIVE_FILE,
            decode_cf=True,
            mask_and_scale=True,
        )

        try:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Compare the native coordinate VARIABLES:
            #
            #   native["lon"]
            #   native["lat"]
            #
            # NOT the HDF5 dimension-scale arrays:
            #
            #   native["longitude"]
            #   native["latitude"]
            #
            # The native file contains both representations.
            # ------------------------------------------------

            native_lon_full = np.asarray(
                native["lon"].values
            )

            native_lat_full = np.asarray(
                native["lat"].values
            )

            native_lon = (
                native_lon_full[
                    LON_START:LON_END + 1
                ]
            )

            native_lat = (
                native_lat_full[
                    LAT_START:LAT_END + 1
                ]
            )

            native_u = np.asarray(
                native["u"]
                .isel(time=0)
                .isel(
                    longitude=slice(
                        LON_START,
                        LON_END + 1,
                    ),
                    latitude=slice(
                        LAT_START,
                        LAT_END + 1,
                    ),
                )
                .values
            )

            native_v = np.asarray(
                native["v"]
                .isel(time=0)
                .isel(
                    longitude=slice(
                        LON_START,
                        LON_END + 1,
                    ),
                    latitude=slice(
                        LAT_START,
                        LAT_END + 1,
                    ),
                )
                .values
            )

            print()
            print("Native subset shapes:")
            print(f"  u: {native_u.shape}")
            print(f"  v: {native_v.shape}")

            print()
            print("OPeNDAP subset shapes:")
            print(f"  u: {subset_u.shape}")
            print(f"  v: {subset_v.shape}")

            # ------------------------------------------------
            # Shape comparison
            # ------------------------------------------------

            shapes_match = (
                native_u.shape == subset_u.shape
                and
                native_v.shape == subset_v.shape
            )

            print()
            print(
                f"Data shapes: "
                f"{'PASS' if shapes_match else 'FAIL'}"
            )

            if not shapes_match:
                raise RuntimeError(
                    "Native and OPeNDAP data "
                    "shapes do not match."
                )

            # ------------------------------------------------
            # Coordinate comparison
            # ------------------------------------------------

            print()
            print(
                "Comparing native and "
                "OPeNDAP coordinates..."
            )

            lon_match, lat_match = (
                compare_coordinates(
                    native_lon,
                    native_lat,
                    lon,
                    lat,
                )
            )

            # ------------------------------------------------
            # u comparison
            # ------------------------------------------------

            u_match = compare_data(
                native_u,
                subset_u,
                "u",
            )

            # ------------------------------------------------
            # v comparison
            # ------------------------------------------------

            v_match = compare_data(
                native_v,
                subset_v,
                "v",
            )

            # ------------------------------------------------
            # Equivalence result
            # ------------------------------------------------

            overall_pass = (
                lon_match
                and lat_match
                and u_match
                and v_match
            )

            print_header(
                "NATIVE vs OPeNDAP EQUIVALENCE"
            )

            print()
            print(
                f"  Longitude : "
                f"{'PASS' if lon_match else 'FAIL'}"
            )

            print(
                f"  Latitude  : "
                f"{'PASS' if lat_match else 'FAIL'}"
            )

            print(
                f"  u values  : "
                f"{'PASS' if u_match else 'FAIL'}"
            )

            print(
                f"  v values  : "
                f"{'PASS' if v_match else 'FAIL'}"
            )

            print()
            print(
                f"  OVERALL   : "
                f"{'PASS' if overall_pass else 'FAIL'}"
            )

        finally:
            native.close()

    finally:
        ds.close()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_header(
        "FINAL OPeNDAP TEST RESULT"
    )

    print()
    print(
        f"Native file size   : "
        f"{native_size:,} bytes"
    )

    print(
        f"Native file size   : "
        f"{format_mib(native_size):.3f} MiB"
    )

    print()
    print(
        f"Subset file size   : "
        f"{total_bytes:,} bytes"
    )

    print(
        f"Subset file size   : "
        f"{format_mib(total_bytes):.3f} MiB"
    )

    if total_bytes > 0:

        reduction = (
            native_size /
            total_bytes
        )

        percentage = (
            100.0 *
            (
                1.0 -
                total_bytes /
                native_size
            )
        )

        print()
        print(
            f"Size reduction     : "
            f"{reduction:.1f}x smaller"
        )

        print(
            f"Transfer reduction : "
            f"{percentage:.2f}%"
        )

    print()
    print("OPeNDAP output:")
    print(f"  {OUTPUT_FILE}")

    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    print()

    if overall_pass:

        print(
            "RESULT: PASS"
        )

        print()
        print(
            "Server-side OPeNDAP subsetting is "
            "validated for OSCAR V2."
        )

    else:

        print(
            "RESULT: FAIL"
        )

        print()
        print(
            "Native and OPeNDAP results are "
            "not fully equivalent."
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Test interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 70)
        print("TEST FAILED")
        print("=" * 70)

        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)