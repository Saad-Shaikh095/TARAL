import netCDF4 as nc4
import os, glob

base = r'd:\OceanEmbed\oceanembed-data\raw\argo'
all_params = set()
sampled = 0
errors = 0

for root, dirs, files in os.walk(base):
    nc_files = [f for f in files if f.endswith('.nc')]
    if nc_files:
        filepath = os.path.join(root, nc_files[0])
        try:
            ds = nc4.Dataset(filepath)
            if 'STATION_PARAMETERS' in ds.variables:
                params = nc4.chartostring(ds['STATION_PARAMETERS'][:]).flatten()
                params = {p.strip() for p in params if p.strip()}
                all_params |= params
            ds.close()
            sampled += 1
        except Exception as e:
            errors += 1

print(f"Sampled {sampled} files, {errors} errors")
print("All unique STATION_PARAMETERS:", sorted(all_params))
