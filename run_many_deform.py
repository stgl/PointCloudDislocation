import subprocess

file = 'data/HSLSurvey101419_utm.csv'
depths = [10, 5, 2, 1]
dip_slips = [0.1, 0.5, 1, 2]
strike_slips = [0.1, 0.5, 1, 2]

for depth in depths:
    for ds in dip_slips:
        print(f'Running {file}, depth = {depth:.1f}, ds = {ds:.2f}...')
        cmd = ['python3', 'deform_point_cloud.py', '-i', file, '--depth', str(depth), '-ds', str(ds), '-o', 'output/dip_slip']
        subprocess.run(cmd)

for depth in depths:
    for ss in strike_slips:
        print(f'Running {file}, depth = {depth:.1f}, ss = {ss:.2f}...')
        cmd = ['python3', 'deform_point_cloud.py', '-i', file, '--depth', str(depth), '-ss', str(ss), '-o', 'output/strike_slip']
        subprocess.run(cmd)
