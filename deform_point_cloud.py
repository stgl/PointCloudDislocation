import os
import argparse
import subprocess
import numpy as np
from disloc import deform_point_cloud_dislocation as deform

parser = argparse.ArgumentParser()
parser.add_argument("--strike", default=0, type=float)
parser.add_argument("--dip", default=89.999, type=float)
parser.add_argument("--depth", default=0.001, type=float)
parser.add_argument("-ss", "--slip_ss", default=0, type=float)
parser.add_argument("-ds", "--slip_ds", default=1.0, type=float)
parser.add_argument(
    "--input", default="data/HSLSurvey101319_utm_thin100.csv"
)
parser.add_argument("--output_dir", default="deformed/")
parser.add_argument("-c", "--convert_to_las", action="store_true")
parser.add_argument("-v", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    output = os.path.join(
        args.output_dir,
        f"hsl_s{args.strike:.2f}_d{args.dip:.2f}_dep{args.depth:.2f}_ss{args.slip_ss:.2f}_ds{args.slip_ds:.2f}.csv",
    )

    if args.v:
        print(f"Reading data from {args.input}...")

    deform(
        args.input,
        output,
        strike=args.strike,
        dip=args.dip,
        depth=args.depth,
        slip_ss=args.slip_ss,
        slip_ds=args.slip_ds,
    )

    if args.v:
        print(f"Wrote output to {output}")

    if args.convert_to_las:
        cmd = [
            "pdal",
            "translate",
            "-i",
            output,
            "-o",
            output.replace(".csv", ".las"),
            "--readers.text.header=X,Y,Z",
        ]
        subprocess.run(cmd)
