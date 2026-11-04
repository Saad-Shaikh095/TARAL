"""
Fast Argo audit: samples ONE file per float directory (not all files).
This gives full schema coverage without scanning 100k+ files.
"""
import netCDF4 as nc4
import os, warnings
warnings.filterwarnings('ignore')

base = r'd:\OceanEmbed\oceanembed-data\raw\argo'

# ── 1. Walk: collect one .nc per float/profiles dir + count totals ───────────
sampled_files = []   # (filepath, month, dac, float_id)
all_nc_count  = 0
months_set, dacs_set, floats_set = set(), set(), set()

for root, dirs, files in os.walk(base):
    nc_files = [f for f in files if f.endswith('.nc')]
    if not nc_files:
        continue
    all_nc_count += len(nc_files)
    parts = root.replace(base, '').lstrip(os.sep).split(os.sep)
    month   = parts[0] if len(parts) > 0 else ''
    dac     = parts[1] if len(parts) > 1 else ''
    float_id= parts[2] if len(parts) > 2 else ''
    months_set.add(month); dacs_set.add(dac); floats_set.add(float_id)
    sampled_files.append((os.path.join(root, nc_files[0]), month, dac, float_id))

print(f"Total .nc files   : {all_nc_count}")
print(f"Float directories sampled: {len(sampled_files)}")

# ── 2. Build unique schemas ───────────────────────────────────────────────────
schemas      = {}   # param_tuple -> {var: info}
param_example= {}   # param_tuple -> example path

for fp, month, dac, float_id in sampled_files:
    try:
        ds = nc4.Dataset(fp)
        if 'STATION_PARAMETERS' not in ds.variables:
            ds.close()
            continue
        raw    = nc4.chartostring(ds['STATION_PARAMETERS'][:]).flatten()
        params = tuple(sorted({p.strip() for p in raw if p.strip()}))

        if params not in schemas:
            schema = {}
            for vname, var in ds.variables.items():
                schema[vname] = {
                    'dtype': str(var.dtype),
                    'dims' : list(var.dimensions),
                }
            schemas[params] = schema
            param_example[params] = fp
        ds.close()
    except Exception as e:
        pass

# ── 3. Print audit ────────────────────────────────────────────────────────────
SEP = "=" * 80
sep = "-" * 80

print()
print(SEP)
print("  ARGO RAW DATASET — FULL AUDIT")
print(SEP)

print(f"""
SUMMARY
  Total .nc files        : {all_nc_count:,}
  Months covered         : {len(months_set)}  ({min(months_set)} → {max(months_set)})
  DACs                   : {len(dacs_set)}   {sorted(dacs_set)}
  Unique float IDs       : {len(floats_set):,}
  Unique variable schemas: {len(schemas)}
""")

print("PATH STRUCTURE")
print(f"  {base}\\<YYYY_MM>\\<dac>\\<float_id>\\profiles\\<PREFIX><float_id>_<cycle>.nc")
print("""
  File prefixes:
    D = Delayed-mode (fully QC'd)
    R = Real-time
    A = Adjusted real-time
""")

for i, (params, schema) in enumerate(schemas.items(), 1):
    print(SEP)
    print(f"  SCHEMA {i} of {len(schemas)}")
    print(f"  STATION_PARAMETERS: {list(params)}")
    print(f"  Example file      : {param_example[params]}")
    print(sep)
    print(f"  {'Variable':<45} {'dtype':<10} Dimensions")
    print(f"  {'-'*43} {'-'*8} {'-'*30}")
    for vname, info in schema.items():
        dims = ', '.join(info['dims'])
        print(f"  {vname:<45} {info['dtype']:<10} ({dims})")

print()
print(SEP)
print("  END OF AUDIT")
print(SEP)
