import numpy as np


def deform_dislocation(
    x,
    y,
    z,
    dip=90.0,
    strike=0.0,
    depth=0.0,
    slip_ss=-1.0,
    slip_ds=0.0,
    origin_x=None,
    origin_y=None,
):
    if depth == 0.0:
        depth = 0.0001
    if np.abs(dip) == 90.0:
        dip = np.sign(dip) * 89.9999

    x0 = np.mean(x) if origin_x is None else origin_x
    y0 = np.mean(y) if origin_y is None else origin_y

    xt = x - x0
    yt = y - y0

    deltarad = np.deg2rad(dip)
    thetarad = np.deg2rad(strike)
    X1p = xt * np.cos(np.pi - thetarad) + yt * np.sin(np.pi - thetarad)
    Zeta = (X1p / depth) - (1 / np.tan(deltarad))
    u1 = (slip_ds / np.pi) * (
        np.cos(deltarad) * np.arctan(Zeta)
        + (np.sin(deltarad) - Zeta * np.cos(deltarad)) / (1 + np.power(Zeta, 2))
    )
    u3 = (-slip_ds / np.pi) * (
        np.sin(deltarad) * np.arctan(Zeta)
        + (np.cos(deltarad) + Zeta * np.sin(deltarad)) / (1 + np.power(Zeta, 2))
    )
    u2 = (slip_ss / np.pi) * (
        np.arctan2(
            Zeta * np.power(np.sin(deltarad), 2),
            (1 - Zeta * np.sin(deltarad) * np.cos(deltarad)),
        )
        + (deltarad - np.sign(deltarad) * np.pi / 2.0)
    )

    u1p = u1 * np.cos(thetarad - np.pi) + u2 * np.sin(thetarad - np.pi)
    u2p = -u1 * np.sin(thetarad - np.pi) + u2 * np.cos(thetarad - np.pi)

    return x + u1p, y + u2p, z + u3


def deform_point_cloud_dislocation(input_filename, output_filename, **kwargs):
    print('Reading input...')
    xyz = np.loadtxt(input_filename, delimiter=",", skiprows=1)

    print('Applying displacements...')
    xd, yd, zd = deform_dislocation(xyz[:, 0], xyz[:, 1], xyz[:, 2], **kwargs)

    print('Saving output...')
    np.savetxt(
        output_filename, np.stack([xd, yd, zd], axis=-1), delimiter=",", fmt="%.2f"
    )

    #return np.stack([xd, yd, zd], axis=-1)


def deform_point_cloud_dislocation_by_point(input_filename, output_filename, **kwargs):
    infile = open(input_filename, 'r')
    outfile = open(output_filename, 'a+')
    for line in infile:
        xyz = list(map(float, line.split(',')))
        xd, yd, zd = deform_dislocation(xyz[0], xyz[1], xyz[2], **kwargs)
        outline = f'{xd:.5f}, {yd:.5f}, {zd:.5f}\n'  
        outfile.write(outline)

    #return np.stack([xd, yd, zd], axis=-1)
