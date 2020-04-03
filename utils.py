import pickle
import numpy as np

from scipy.ndimage.morphology import binary_erosion

from disloc import deform_dislocation

def load_and_mask_results(filename, MAX_RESIDUAL=0.05):
    (x,y,ux,uy,uz,residual) = pickle.load(open(filename,'rb'))

    mask_residual = np.abs(residual) <= MAX_RESIDUAL
    mask_boundary = ~np.isnan(residual)
    mask_boundary = binary_erosion(mask_boundary, iterations = 2)
    mask = mask_residual * mask_boundary
    
    ux[~mask] = np.nan
    uy[~mask] = np.nan
    uz[~mask] = np.nan
    
    return x, y, ux, uy, uz, residual, mask

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
    
def calculate_displacements(shape, strike=0, dip=89.999, depth=0.001, dip_slip=1, strike_slip=0, theta=None):
    xv = np.linspace(0, shape[1], num=shape[1])
    yv = np.linspace(0, shape[0], num=shape[0])
    x, y = np.meshgrid(xv, yv)
    z = np.zeros_like(x)
    ux, uy, uz = deform_dislocation(x, y, z, strike=strike, dip=dip, depth=depth, slip_ds=dip_slip, slip_ss=strike_slip)
    if theta is None:
        return uz
    else:
        u = transform_displacements(ux, uy, theta=theta)
        return u