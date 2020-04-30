import pickle
import numpy as np

from scipy.ndimage.morphology import binary_erosion

from disloc import deform_dislocation

ORIGIN_X = 321660
ORIGIN_Y = 4164942


def load_and_mask_results(filename, iterations=2, MAX_RESIDUAL=0.05):
    (x, y, ux, uy, uz, residual) = pickle.load(open(filename, "rb"))

    mask_residual = np.abs(residual) <= MAX_RESIDUAL
    mask_boundary = ~np.isnan(residual)
    mask_boundary = binary_erosion(mask_boundary, iterations=iterations)
    mask = mask_residual * mask_boundary

    ux[~mask] = np.nan
    uy[~mask] = np.nan
    uz[~mask] = np.nan

    return x, y, ux, uy, uz, residual, mask

def apply_water_mask(u, dx=50):
    mask = np.zeros_like(u).astype(bool)
    if dx == 25:
        mask[20:39, 36:45] = True
    if dx == 50:
        mask[12:19, 18:25] = True
    if dx == 100:
        mask[6:10, 9:13] = True
    if dx == 150:
        mask[4:7, 6:9] = True
    if dx == 200:
        mask[4:6, 4:6] = True
    umasked = u.copy()
    umasked[mask] = np.nan
    return umasked

def calc_xyz_rmses(ux, uy, uz, uxtrue, uytrue, uztrue):
    rmse_x = np.sqrt(np.nanmean((ux - uxtrue) ** 2))
    rmse_y = np.sqrt(np.nanmean((uy - uytrue) ** 2))
    rmse_z = np.sqrt(np.nanmean((uz - uztrue) ** 2))
    uh = np.sqrt(ux ** 2 + uy ** 2)
    uhtrue = np.sqrt(uxtrue ** 2 + uytrue ** 2)
    rmse_h = np.sqrt(np.nanmean((uh - uhtrue) ** 2))
    return rmse_x, rmse_y, rmse_z, rmse_h
    
def transform_displacements(ux, uy, theta=0):
    if theta == 0:
        return uy
    if theta == 90:
        return ux
    else:
        # TODO: Fix this!
        alpha = (90 - (360 - theta)) * np.pi / 180
        ualpha = -ux * np.cos(alpha) + uy * np.sin(alpha)
        return ualpha

def calculate_displacements_profile(
    strike=0, dip=89.999, depth=0.001, dip_slip=1, strike_slip=0, xlim=None, ylim=None
):
    xmin, xmax = xlim
    ymin, ymax = ylim
    xv = np.linspace(xmin, xmax, num=int(xmax - xmin))
    yv = np.linspace(ymin, ymax, num=int(ymax - ymin))
    x, y = np.meshgrid(xv, yv)
    z = np.zeros_like(x)
    ux, uy, uz = deform_dislocation(
        x,
        y,
        z,
        strike=strike,
        dip=dip,
        depth=depth,
        slip_ds=dip_slip,
        slip_ss=strike_slip,
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
    )
    return ux - x, uy - y, uz

def disloc_model_from_coords_and_mask(x, y, mask, **kwargs):
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()
    zz = np.zeros_like(xx)
    xd, yd, uzm = deform_dislocation(xx, yy, zz, **kwargs)
    uxm = xd - xx
    uym = yd - yy
    uxm = uxm.reshape(mask.shape)
    uym = uym.reshape(mask.shape)
    uzm = uzm.reshape(mask.shape)
    uxm[~mask] = np.nan
    uym[~mask] = np.nan
    uzm[~mask] = np.nan
    return uxm, uym, uzm

def rmse(ytrue, yhat):
    return np.sqrt(np.nanmean((yhat - ytrue) ** 2))

def rotate(x, y, theta):
    theta *= np.pi / 180
    u = x * np.cos(theta) - y * np.sin(theta)
    v = x * np.sin(theta) + y * np.cos(theta)
    return u, v
