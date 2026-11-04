from pathlib import Path
import csv
import json
import hashlib
import xarray as xr


# ============================================================
# OSCAR VARIABLE-SIGNATURE INVESTIGATION
# ============================================================

RAW_ROOT = Path(
    
    r"D:\OceanEmbed\oceanembed-data\raw\oscar"
)

AUDIT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\processed\oscar_audit"
)

INVENTORY = AUDIT_DIR / "oscar_file_inventory.csv"

OUTPUT_TXT = AUDIT_DIR / "oscar_variable_signature_investigation.txt"
OUTPUT_JSON = AUDIT_DIR / "oscar_variable_signature_investigation.json"


# ============================================================
# HELPERS
# ============================================================

def normalize(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    return value


def clean_attrs(attrs):
    return {
        str(k): normalize(v)
        for k, v in attrs.items()
    }


def inspect_file(path):

    result = {
        "path": str(path),
        "variables": {},
        "coordinates": {},
        "dimensions": {},
        "dataset_attributes": {},
    }

    with xr.open_dataset(
        path,
        decode_cf=True,
        mask_and_scale=True,
    ) as ds:

        result["dimensions"] = {
            str(k): int(v)
            for k, v in ds.sizes.items()
        }

        result["dataset_attributes"] = clean_attrs(
            dict(ds.attrs)
        )

        for name in ds.data_vars:

            var = ds[name]

            result["variables"][name] = {
                "dims": list(var.dims),
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "attrs": clean_attrs(
                    dict(var.attrs)
                ),
            }

        for name in ds.coords:

            coord = ds[name]

            result["coordinates"][name] = {
                "dims": list(coord.dims),
                "shape": list(coord.shape),
                "dtype": str(coord.dtype),
                "attrs": clean_attrs(
                    dict(coord.attrs)
                ),
            }

    return result


def scientific_variable_signature(info):

    """
    Signature WITHOUT metadata attributes.

    This is the important comparison.

    If two files differ only in metadata, they should have
    the same scientific structure here.
    """

    payload = {
        "dimensions": info["dimensions"],
        "variables": {},
        "coordinates": {},
    }

    for name, var in sorted(
        info["variables"].items()
    ):

        payload["variables"][name] = {
            "dims": var["dims"],
            "shape": var["shape"],
            "dtype": var["dtype"],
        }

    for name, coord in sorted(
        info["coordinates"].items()
    ):

        payload["coordinates"][name] = {
            "dims": coord["dims"],
            "shape": coord["shape"],
            "dtype": coord["dtype"],
        }

    raw = json.dumps(
        payload,
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def attribute_signature(info):

    payload = {
        "dataset_attributes":
            info["dataset_attributes"],

        "variables": {
            name: var["attrs"]
            for name, var
            in sorted(
                info["variables"].items()
            )
        },

        "coordinates": {
            name: coord["attrs"]
            for name, coord
            in sorted(
                info["coordinates"].items()
            )
        },
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# LOAD INVENTORY
# ============================================================

if not INVENTORY.exists():

    raise FileNotFoundError(
        f"Inventory not found:\n{INVENTORY}"
    )


rows = []

with INVENTORY.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        if row.get("opened", "").lower() == "true":

            rows.append(row)


print()
print("=" * 72)
print("OSCAR VARIABLE-SIGNATURE INVESTIGATION")
print("=" * 72)
print()

print(
    "Files available from audit inventory:",
    len(rows),
)

# ============================================================
# GROUP BY EXISTING VARIABLE SIGNATURE
# ============================================================

groups = {}

for row in rows:

    signature = row.get(
        "variable_signature",
        "",
    )

    if not signature:
        continue

    groups.setdefault(
        signature,
        []
    ).append(row)


print(
    "Existing variable signatures:",
    len(groups),
)

print()

# ============================================================
# SELECT REPRESENTATIVE FILES
# ============================================================

representatives = []

for signature, group in groups.items():

    first = group[0]

    path = Path(
        first["full_path"]
    )

    representatives.append({
        "existing_signature": signature,
        "file_count": len(group),
        "representative_file": str(path),
    })


# ============================================================
# INSPECT REPRESENTATIVES
# ============================================================

investigations = []

for item in representatives:

    path = Path(
        item["representative_file"]
    )

    print(
        "Inspecting:",
        path.name,
    )

    info = inspect_file(path)

    sci_sig = scientific_variable_signature(
        info
    )

    attr_sig = attribute_signature(
        info
    )

    item["scientific_signature"] = sci_sig
    item["attribute_signature"] = attr_sig
    item["inspection"] = info

    investigations.append(item)


# ============================================================
# COMPARE SCIENTIFIC STRUCTURE
# ============================================================

scientific_signatures = {
    item["scientific_signature"]
    for item in investigations
}


attribute_signatures = {
    item["attribute_signature"]
    for item in investigations
}


if len(scientific_signatures) == 1:

    scientific_consistency = True

else:

    scientific_consistency = False


if len(attribute_signatures) > 1:

    metadata_difference_exists = True

else:

    metadata_difference_exists = False


# ============================================================
# DETAILED COMPARISON
# ============================================================

comparison = []

if len(investigations) >= 2:

    base = investigations[0]["inspection"]

    for other_item in investigations[1:]:

        other = other_item["inspection"]

        differences = []

        # --------------------------------------------
        # Dimensions
        # --------------------------------------------

        if (
            base["dimensions"]
            != other["dimensions"]
        ):

            differences.append(
                "DIMENSIONS"
            )

        # --------------------------------------------
        # Variables
        # --------------------------------------------

        base_vars = set(
            base["variables"]
        )

        other_vars = set(
            other["variables"]
        )

        if base_vars != other_vars:

            differences.append(
                "VARIABLE_NAMES"
            )

        # --------------------------------------------
        # Variable scientific properties
        # --------------------------------------------

        common_vars = (
            base_vars
            & other_vars
        )

        for name in sorted(
            common_vars
        ):

            b = base["variables"][name]
            o = other["variables"][name]

            for field in [
                "dims",
                "shape",
                "dtype",
            ]:

                if b[field] != o[field]:

                    differences.append(
                        f"VARIABLE_{name}_{field.upper()}"
                    )

        # --------------------------------------------
        # Variable attributes
        # --------------------------------------------

        attribute_differences = []

        for name in sorted(
            common_vars
        ):

            b_attrs = base[
                "variables"
            ][name]["attrs"]

            o_attrs = other[
                "variables"
            ][name]["attrs"]

            if b_attrs != o_attrs:

                attribute_differences.append(
                    name
                )

        # --------------------------------------------
        # Coordinate scientific properties
        # --------------------------------------------

        base_coords = set(
            base["coordinates"]
        )

        other_coords = set(
            other["coordinates"]
        )

        if base_coords != other_coords:

            differences.append(
                "COORDINATE_NAMES"
            )

        for name in sorted(
            base_coords
            & other_coords
        ):

            b = base[
                "coordinates"
            ][name]

            o = other[
                "coordinates"
            ][name]

            for field in [
                "dims",
                "shape",
                "dtype",
            ]:

                if b[field] != o[field]:

                    differences.append(
                        f"COORDINATE_{name}_{field.upper()}"
                    )

        # --------------------------------------------
        # Coordinate attributes
        # --------------------------------------------

        coordinate_attribute_differences = []

        for name in sorted(
            base_coords
            & other_coords
        ):

            b_attrs = base[
                "coordinates"
            ][name]["attrs"]

            o_attrs = other[
                "coordinates"
            ][name]["attrs"]

            if b_attrs != o_attrs:

                coordinate_attribute_differences.append(
                    name
                )

        # --------------------------------------------
        # Dataset attributes
        # --------------------------------------------

        dataset_attrs_differ = (
            base["dataset_attributes"]
            != other["dataset_attributes"]
        )

        comparison.append({

            "base_file":
                investigations[0][
                    "representative_file"
                ],

            "other_file":
                other_item[
                    "representative_file"
                ],

            "scientific_differences":
                differences,

            "variable_attribute_differences":
                attribute_differences,

            "coordinate_attribute_differences":
                coordinate_attribute_differences,

            "dataset_attributes_differ":
                dataset_attrs_differ,

        })


# ============================================================
# FINAL INTERPRETATION
# ============================================================

if scientific_consistency:

    if metadata_difference_exists:

        conclusion = (
            "GREEN CONFIRMED: The two existing "
            "variable signatures differ only because "
            "of metadata/attributes. The scientific "
            "variable structure, dimensions, shapes, "
            "and dtypes are consistent."
        )

    else:

        conclusion = (
            "GREEN CONFIRMED: The files have "
            "consistent scientific structure."
        )

else:

    conclusion = (
        "YELLOW: The variable signatures represent "
        "different scientific structures. Further "
        "investigation is required before permanently "
        "locking OSCAR as GREEN."
    )


# ============================================================
# REPORT
# ============================================================

report = {

    "investigation": {
        "timestamp_utc":
            __import__(
                "datetime"
            ).datetime.now(
                __import__(
                    "datetime"
                ).timezone.utc
            ).isoformat(),

        "raw_root":
            str(RAW_ROOT),

        "audit_inventory":
            str(INVENTORY),

        "raw_data_modified":
            False,
    },

    "existing_variable_signatures": len(
        groups
    ),

    "scientific_signatures": len(
        scientific_signatures
    ),

    "attribute_signatures": len(
        attribute_signatures
    ),

    "scientific_structure_consistent":
        scientific_consistency,

    "metadata_difference_exists":
        metadata_difference_exists,

    "representatives":
        investigations,

    "comparisons":
        comparison,

    "conclusion":
        conclusion,
}


# ============================================================
# WRITE JSON
# ============================================================

with OUTPUT_JSON.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# WRITE HUMAN-READABLE REPORT
# ============================================================

with OUTPUT_TXT.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "============================================================\n"
    )

    f.write(
        "OSCAR VARIABLE-SIGNATURE INVESTIGATION\n"
    )

    f.write(
        "============================================================\n\n"
    )

    f.write(
        f"Existing variable signatures: "
        f"{len(groups)}\n"
    )

    f.write(
        f"Scientific signatures: "
        f"{len(scientific_signatures)}\n"
    )

    f.write(
        f"Attribute signatures: "
        f"{len(attribute_signatures)}\n\n"
    )

    f.write(
        "============================================================\n"
    )

    f.write(
        "REPRESENTATIVE FILES\n"
    )

    f.write(
        "============================================================\n\n"
    )

    for i, item in enumerate(
        investigations,
        start=1,
    ):

        f.write(
            f"GROUP {i}\n"
        )

        f.write(
            f"Files in group: "
            f"{item['file_count']}\n"
        )

        f.write(
            f"Representative: "
            f"{item['representative_file']}\n"
        )

        f.write(
            f"Original audit signature: "
            f"{item['existing_signature']}\n"
        )

        f.write(
            f"Scientific signature: "
            f"{item['scientific_signature']}\n"
        )

        f.write(
            f"Attribute signature: "
            f"{item['attribute_signature']}\n\n"
        )

        info = item["inspection"]

        f.write(
            "Dimensions:\n"
        )

        for key, value in info[
            "dimensions"
        ].items():

            f.write(
                f"  {key}: {value}\n"
            )

        f.write("\nVariables:\n")

        for name, var in info[
            "variables"
        ].items():

            f.write(
                f"  {name}\n"
            )

            f.write(
                f"    dims: {var['dims']}\n"
            )

            f.write(
                f"    shape: {var['shape']}\n"
            )

            f.write(
                f"    dtype: {var['dtype']}\n"
            )

            f.write(
                f"    attrs: "
                f"{json.dumps(var['attrs'], default=str)}\n"
            )

        f.write("\nCoordinates:\n")

        for name, coord in info[
            "coordinates"
        ].items():

            f.write(
                f"  {name}\n"
            )

            f.write(
                f"    dims: {coord['dims']}\n"
            )

            f.write(
                f"    shape: {coord['shape']}\n"
            )

            f.write(
                f"    dtype: {coord['dtype']}\n"
            )

            f.write(
                f"    attrs: "
                f"{json.dumps(coord['attrs'], default=str)}\n"
            )

        f.write("\n")

    f.write(
        "============================================================\n"
    )

    f.write(
        "COMPARISON\n"
    )

    f.write(
        "============================================================\n\n"
    )

    for item in comparison:

        f.write(
            f"Base: {item['base_file']}\n"
        )

        f.write(
            f"Other: {item['other_file']}\n"
        )

        f.write(
            "Scientific differences:\n"
        )

        if item[
            "scientific_differences"
        ]:

            for diff in item[
                "scientific_differences"
            ]:

                f.write(
                    f"  - {diff}\n"
                )

        else:

            f.write(
                "  NONE\n"
            )

        f.write(
            "Variable attribute differences:\n"
        )

        if item[
            "variable_attribute_differences"
        ]:

            for diff in item[
                "variable_attribute_differences"
            ]:

                f.write(
                    f"  - {diff}\n"
                )

        else:

            f.write(
                "  NONE\n"
            )

        f.write(
            "Coordinate attribute differences:\n"
        )

        if item[
            "coordinate_attribute_differences"
        ]:

            for diff in item[
                "coordinate_attribute_differences"
            ]:

                f.write(
                    f"  - {diff}\n"
                )

        else:

            f.write(
                "  NONE\n"
            )

        f.write(
            "Dataset attributes differ: "
            f"{item['dataset_attributes_differ']}\n\n"
        )

    f.write(
        "============================================================\n"
    )

    f.write(
        "FINAL CONCLUSION\n"
    )

    f.write(
        "============================================================\n\n"
    )

    f.write(
        conclusion
        + "\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print()
print("=" * 72)
print("INVESTIGATION COMPLETE")
print("=" * 72)
print()

print(
    "Existing variable signatures:",
    len(groups),
)

print(
    "Scientific signatures:",
    len(scientific_signatures),
)

print(
    "Attribute signatures:",
    len(attribute_signatures),
)

print(
    "Scientific structure consistent:",
    scientific_consistency,
)

print(
    "Metadata differences detected:",
    metadata_difference_exists,
)

print()
print("CONCLUSION:")
print(conclusion)

print()
print("TXT:")
print(OUTPUT_TXT)

print()
print("JSON:")
print(OUTPUT_JSON)

print()
print("=" * 72)