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

    se = slip_ds  # Negative is reverse motion.
    ss = slip_ss  # Negative is right-lateral strike-slip motion

    d = depth if depth is not 0.0 else 0.0001
    delta = dip if np.abs(dip) < 90.0 else np.sign(dip) * 89.9999
    theta = strike  # degrees

    x0 = np.mean(x) if origin_x is None else origin_x
    y0 = np.mean(y) if origin_y is None else origin_y

    x1 = x - x0
    x2 = y - y0

    deltarad = np.deg2rad(delta)
    thetarad = np.deg2rad(theta)
    X1p = x1 * np.cos(np.pi - thetarad) + x2 * np.sin(np.pi - thetarad)
    Zeta = (X1p / d) - (1 / np.tan(deltarad))
    U1 = (se / np.pi) * (
        np.cos(deltarad) * np.arctan(Zeta)
        + (np.sin(deltarad) - Zeta * np.cos(deltarad)) / (1 + np.power(Zeta, 2))
    )
    U2 = (-se / np.pi) * (
        np.sin(deltarad) * np.arctan(Zeta)
        + (np.cos(deltarad) + Zeta * np.sin(deltarad)) / (1 + np.power(Zeta, 2))
    )
    U3 = (ss / np.pi) * (
        np.arctan2(
            Zeta * np.power(np.sin(deltarad), 2),
            (1 - Zeta * np.sin(deltarad) * np.cos(deltarad)),
        )
        + (deltarad - np.sign(deltarad) * np.pi / 2.0)
    )

    U1p = U1 * np.cos(thetarad - np.pi) + U3 * np.sin(thetarad - np.pi)
    U2p = -U1 * np.sin(thetarad - np.pi) + U3 * np.cos(thetarad - np.pi)
    U3p = U2

    return x + U1p, y + U2p, z + U3p


def deform_point_cloud_dislocation(input_filename, output_filename, **kwargs):
    xyz = np.loadtxt(input_filename, delimiter=',', skiprows=1)

    xd, yd, zd = deform_dislocation(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)
    
    np.savetxt(output_filename, np.stack([xd, yd, zd]), delimiter=',')
