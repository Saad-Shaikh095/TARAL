import netCDF4 as nc4
import os, glob, json
from collections import defaultdict

base = r'd:\OceanEmbed\oceanembed-data\raw\argo'

# ── 1. Collect all .nc file paths ───────────────────────────────────────────
all_files = []
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.nc'):
            all_files.append(os.path.join(root, f))
print(f"Total .nc files: {len(all_files)}\n")

# ── 2. Collect unique variable schemas keyed by frozenset(STATION_PARAMETERS) ─
schemas = {}   # param_key -> {variable_name: {dtype, dims, shape_template}}
param_examples = {}  # param_key -> example file

for fp in all_files:
    try:
        ds = nc4.Dataset(fp)
        if 'STATION_PARAMETERS' not in ds.variables:
            ds.close()
            continue
        raw = nc4.chartostring(ds['STATION_PARAMETERS'][:]).flatten()
        params = tuple(sorted({p.strip() for p in raw if p.strip()}))
        key = params

        if key not in schemas:
            schema = {}
            for vname, var in ds.variables.items():
                schema[vname] = {
                    'dtype': str(var.dtype),
                    'dimensions': list(var.dimensions),
                    'shape': list(var.shape),
                }
            schemas[key] = schema
            param_examples[key] = fp
        ds.close()
    except Exception as e:
        print(f"ERROR {fp}: {e}")

# ── 3. Print the audit ───────────────────────────────────────────────────────
print("=" * 80)
print("ARGO RAW DATASET AUDIT")
print("=" * 80)

print(f"\n{'─'*40}")
print(f"TOTAL FILES : {len(all_files)}")
# summarise directory structure
months = set()
dacs   = set()
floats = set()
for fp in all_files:
    parts = fp.replace(base, '').lstrip(os.sep).split(os.sep)
    if len(parts) >= 1: months.add(parts[0])
    if len(parts) >= 2: dacs.add(parts[1])
    if len(parts) >= 3: floats.add(parts[2])

print(f"MONTHS      : {len(months)}  ({min(months)} → {max(months)})")
print(f"DACs        : {len(dacs)}  {sorted(dacs)}")
print(f"UNIQUE FLOATS: {len(floats)}")

print(f"\n{'─'*40}")
print("PATH STRUCTURE")
print(f"{'─'*40}")
print(f"  {base}\\<YYYY_MM>\\<dac>\\<float_id>\\profiles\\<[DAR]float_cycle.nc>")
print()
print("  File prefix convention:")
print("   D = Delayed mode (QC'd)")
print("   R = Real-time mode")
print("   A = Adjusted real-time")

print(f"\n{'─'*40}")
print(f"UNIQUE PARAMETER SCHEMAS FOUND: {len(schemas)}")
print(f"{'─'*40}")

for i, (key, schema) in enumerate(schemas.items(), 1):
    print(f"\n── Schema {i}: STATION_PARAMETERS = {list(key)}")
    print(f"   Example file: {param_examples[key]}")
    print(f"\n   {'Variable':<40} {'dtype':<12} {'dimensions'}")
    print(f"   {'-'*38} {'-'*10} {'-'*40}")
    for vname, info in schema.items():
        dims_str = str(info['dimensions'])
        print(f"   {vname:<40} {info['dtype']:<12} {dims_str}")

print("\n" + "=" * 80)
print("END OF AUDIT")
print("=" * 80)
