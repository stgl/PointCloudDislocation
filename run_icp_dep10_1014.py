import os
import pickle
import subprocess
import numpy as np
import sys
sys.path.append('/home/rmsare/PML/')
import pml
#import PML as pml


#depths = [10, 5, 2, 1]
#dip_slips = [0.1, 0.5, 1, 2]
#strike_slips = [0.1, 0.5, 1, 2]

INPUT_FILENAME = 'data/HSLSurvey101319_utm.las'
TARGET_STRING = 'dep10.0'
base = INPUT_FILENAME.split('/')[-1].split('.')[0].replace('13', '14')

def list_files(base='', substr=''):
    files1 = os.listdir('output/dip_slip/')
    files1 = ['output/dip_slip/' + f for f in files1 if '.las' in f]
    files2 = os.listdir('output/strike_slip/')
    files2 = ['output/strike_slip/' + f for f in files2 if '.las' in f]
    files = files1 + files2
    files = [f for f in files if base in f and substr in f]
    return files


dx_window = 50
dy_window = dx_window
dx = 25.0
dy = dx
buffer_fraction = 0.1
use_dask = True

#fixed = pml.read_file(INPUT_FILENAME)
fixed = INPUT_FILENAME 
files = list_files(base=base, substr=TARGET_STRING)
print(f'Found {files}')

for moving in files:
    output = moving.replace("output", "results").replace(
        ".las", f"_ICP_w{dx_window:.2f}_str{dx:.2f}.pkl"
    )
    if not os.path.exists(output):
        print(f"Writing output to {output}")
        
        bounds = pml.get_bounds(fixed)
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
            use_dask=use_dask,
        )


        pickle.dump((x, y, ux, uy, uz, residuals), open(output, "wb"))
    else:
        print(f'{output} exists. Skipping.' )
    files = list_files(base=base, substr=TARGET_STRING)
