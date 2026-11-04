"""
Quick Argo audit — stops as soon as all unique schemas are found.
Scans at most one file per DAC per month until no new schemas appear.
"""
import netCDF4 as nc4
import os, warnings
warnings.filterwarnings('ignore')

base = r'd:\OceanEmbed\oceanembed-data\raw\argo'

schemas      = {}   # param_tuple -> {var: (dtype, dims)}
param_example= {}
files_read   = 0

# Walk only top-level month dirs, then one file per DAC subdir
for month in sorted(os.listdir(base)):
    month_path = os.path.join(base, month)
    if not os.path.isdir(month_path): continue
    for dac in sorted(os.listdir(month_path)):
        dac_path = os.path.join(month_path, dac)
        if not os.path.isdir(dac_path): continue
        # grab first .nc found anywhere under this dac
        for root, _, files in os.walk(dac_path):
            nc_files = [f for f in files if f.endswith('.nc')]
            if not nc_files: continue
            fp = os.path.join(root, nc_files[0])
            try:
                ds = nc4.Dataset(fp)
                files_read += 1
                if 'STATION_PARAMETERS' in ds.variables:
                    raw    = nc4.chartostring(ds['STATION_PARAMETERS'][:]).flatten()
                    params = tuple(sorted({p.strip() for p in raw if p.strip()}))
                    if params not in schemas:
                        schema = {vn: (str(v.dtype), list(v.dimensions))
                                  for vn, v in ds.variables.items()}
                        schemas[params] = schema
                        param_example[params] = fp
                ds.close()
            except:
                pass
            break  # one file per dac is enough

print(f"Files read   : {files_read}")
print(f"Schemas found: {len(schemas)}\n")

SEP = "=" * 78
sep = "-" * 78

print(SEP)
print("  ARGO RAW DATASET — AUDIT")
print(SEP)

print(f"""
PATH STRUCTURE
  {base}\\<YYYY_MM>\\<dac>\\<float_id>\\profiles\\<PREFIX><float_id>_<cycle>.nc

  Prefixes:  D = Delayed-mode (QC'd)   R = Real-time   A = Adjusted real-time

DIRECTORY LAYOUT
  Months : 2012_01 → 2024_12  (156 month folders)
  DACs   : aoml, bodc, coriolis, csio, csiro, incois, nmdis
  Floats : one sub-folder per float WMO ID, contains a profiles/ sub-folder
""")

for i, (params, schema) in enumerate(schemas.items(), 1):
    print(SEP)
    print(f"  SCHEMA {i}/{len(schemas)}  —  STATION_PARAMETERS: {list(params)}")
    print(f"  Example: {param_example[params]}")
    print(sep)
    print(f"  {'Variable':<44} {'dtype':<10} Dimensions")
    print(f"  {'-'*42} {'-'*8} {'-'*24}")
    for vname, (dtype, dims) in schema.items():
        print(f"  {vname:<44} {dtype:<10} ({', '.join(dims)})")

print()
print(SEP)
print("  END OF AUDIT")
print(SEP)
