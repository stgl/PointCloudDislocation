"""
Convenience functions for plotting point cloud features
"""

import numpy as np
import pandas as pd

from copy import copy

from itertools import combinations
from skimage.color import lab2rgb

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt

def initialize_plot_settings():
    plt.style.use('ggplot')
    matplotlib.rcParams['figure.figsize'] = (8, 8)
    matplotlib.rcParams['font.size'] = 14
    matplotlib.rcParams['axes.labelsize'] = 18
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14
    matplotlib.rcParams['legend.title_fontsize'] = 14
    matplotlib.rcParams['legend.fontsize'] = 14

    matplotlib.rcParams['axes.labelcolor'] ='k'

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.grid'] = False
    
    matplotlib.rcParams['axes.facecolor'] = 'w'
    matplotlib.rcParams['axes.edgecolor'] = 'k'
    matplotlib.rcParams['xtick.color'] = 'k'
    matplotlib.rcParams['ytick.color'] = 'k'
    
    matplotlib.rcParams['legend.frameon'] = False
    
    matplotlib.rcParams['savefig.dpi'] = 300 
    matplotlib.rcParams['savefig.pad_inches'] = 0
    
def plot_results(x, y, ux, uy, uz, depth, ds, strike):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    im = ax[0].imshow(ux, vmin=-1, vmax=1, origin='lower', extent=(np.min(x),np.max(x),np.min(y),np.max(y)))
    plt.colorbar(im, label='Displacement $u_x$ [m]', ax=ax[0], orientation='horizontal', shrink=0.5)

    im = ax[1].imshow(uy, vmin=-1, vmax=1, origin='lower', extent=(np.min(x),np.max(x),np.min(y),np.max(y)))
    plt.colorbar(im, label='Displacement $u_y$ [m]', ax=ax[1], orientation='horizontal', shrink=0.5)
    ax[1].set_axis_off()

    im = ax[2].imshow(uz, vmin=-1, vmax=1, origin='lower', extent=(np.min(x),np.max(x),np.min(y),np.max(y)))
    plt.colorbar(im, label='Displacement $u_z$ [m]', ax=ax[2], orientation='horizontal', shrink=0.5)
    ax[2].set_axis_off()
    
    plt.suptitle(f'Displacements: dip slip = {ds:.2f} m, depth = {depth:.0f} m, strike = {strike:.0f} deg, dip = 90 deg', fontsize=16)
    
def plot_displacement_profile(us, labels, ylabel='Displacement $u_z$ [m]', legend_title='Dip slip', dx=25, strike=None, dip_slip=None, strike_slip=None):
    ij = []
    j = 0
    for i in np.arange(0, us[0].shape[0], dtype=int):
        ij.append([i, j])
        j += 1
        if j > us[0].shape[1]:
            break
    ij = np.array(ij)

    plt.figure(figsize=(10, 5))

    markers = ['ko', 'ks', 'k^']
    for u, m, label in zip(us, markers, labels):
        plt.plot(u[ij[:, 0], ij[:, 1]], m, ms=5, mec='w', ls='None', label=label)
    
    plt.xticks([10, 20, 30, 40])
    xt = np.array(plt.xticks()[0])
    plt.gca().set_xticklabels(dx * xt)
    ymax = np.nanmax(u[ij[:, 0], ij[:, 1]])
    plt.ylim([-1.05 * ymax, 1.05 * ymax])
    
    plt.xlabel('Distance [m]')
    plt.ylabel(ylabel)
    plt.legend(title=legend_title)
    plt.gcf().patch.set_facecolor('w')
    
def plot_displacement_profile_resolution(us, labels, markers, ylabel='Displacement $u_z$ [m]', legend_title='Dip slip', dxs=[5, 12.5, 25, 50]):
    fig, axes = plt.subplots(len(us), 1, figsize=(6, 3 * len(us)))
    for u, ax, m, dx, label in zip(us, axes, markers, dxs, labels):
        ij = []
        j = 0
        for i in np.arange(0, u.shape[0], dtype=int):
            ij.append([i, j])
            j += 1
            if j > u.shape[1]:
                break
        ij = np.array(ij)
        
        xp = dx * np.arange(ij.shape[0])
        up = u[ij[:, 0], ij[:, 1]]
        mask = np.isfinite(up)
        ax.plot(xp[mask], up[mask], m, mec='k', ls='None')
        ax.set_title(f'Stride {dx:.0f} m')
    
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    
    axes[-1].set_xlabel('Distance [m]')
    axes[2].set_ylabel(ylabel)