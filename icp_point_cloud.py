import os
import argparse
import numpy as np
import sys

sys.path.append("/home/rmsare/PML/")
import pml
import numpy as np
import pickle as p


parser = argparse.ArgumentParser()
parser.add_argument("-dxw", "--dx_window", default=50.0, type=float)
parser.add_argument("-dyw", "--dy_window", default=50.0, type=float)
parser.add_argument("-dx", default=25.0, type=float)
parser.add_argument("-dy", default=25.0, type=float)
parser.add_argument("-b", "--buffer_fraction", default=0.1, type=float)

parser.add_argument(
    "--fixed", default="/scratch/rmsare/pointclouds/HSLSurvey101319_utm_thin100.csv"
)
parser.add_argument(
    "--moving",
    default="/scratch/rmsare/output/hsl_s0.00_d90.00_dep0.00_ss0.00_ds10.00.las",
)
parser.add_argument("--output", default=None)
parser.add_argument("-v", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.v:
        print(f"Reading fixed pc from {args.fixed}...")
        print(f"Reading moving pc from {args.moving}...")

    print("Setting ICP params...")
    fixed = args.fixed
    moving = args.moving

    bounds = pml.get_bounds(fixed)
    dx = args.dx
    dy = args.dy
    dx_window = args.dx_window
    dy_window = args.dy_window
    buffer_fraction = args.buffer_fraction
    x = np.arange(bounds[0][0], bounds[0][1], dx)
    y = np.arange(bounds[1][0], bounds[1][1], dy)

    ux, uy, uz, residuals = pml.icp_tile(
        fixed,
        moving,
        x,
        y,
        buffer_fraction=buffer_fraction,
        dx_window=dx_window,
        dy_window=dy_window,
        num_trials=5,
#        debug=True,
        use_dask=False,
    )

    if args.output is None:
        args.output = args.moving.replace("output", "icp").replace(
            ".las", f"._ICP_w{dx_window::.2f}_str{dx:.2f}.pkl"
        )

    p.dump((x, y, ux, uy, uz, residuals), open(output, "wb"))

    if args.v:
        print(f"Wrote output to {output}")
