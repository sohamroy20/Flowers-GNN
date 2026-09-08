"""Print the structure of a RIGNO NetCDF dataset.

Everything downstream depends on the actual variable names, shapes, and
whether connectivity is present, so look before writing a loader.
"""

import argparse
import numpy as np
import xarray as xr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    ds = xr.open_dataset(args.path)
    print("=" * 70)
    print(ds)
    print("=" * 70)

    print("\nDIMENSIONS")
    for k, v in ds.sizes.items():
        print(f"  {k:20s} {v}")

    print("\nVARIABLES")
    for name, var in ds.variables.items():
        print(f"  {name:20s} {str(var.dims):40s} {str(var.shape):20s} {var.dtype}")

    print("\nATTRIBUTES")
    for k, v in ds.attrs.items():
        print(f"  {k}: {v}")

    # Does it ship mesh connectivity, or is it a bare point cloud?
    conn = [n for n in ds.variables
            if any(t in n.lower() for t in ("cell", "tri", "elem", "conn", "face"))]
    print(f"\nconnectivity-like variables: {conn if conn else 'NONE — point cloud'}")

    coords = [n for n in ds.variables
              if any(t in n.lower() for t in ("coord", "pos", "point", "node", "x", "y"))]
    print(f"coordinate-like variables  : {coords}")

    # Value ranges on the main field, for normalization planning.
    for name, var in ds.variables.items():
        if var.ndim >= 3:
            a = np.asarray(var[0] if var.shape[0] > 1 else var)
            print(f"\n{name}: shape {var.shape}")
            print(f"  sample min {np.nanmin(a):.5f}  max {np.nanmax(a):.5f}  "
                  f"mean {np.nanmean(a):.5f}  NaNs {int(np.isnan(a).sum())}")


if __name__ == "__main__":
    main()
