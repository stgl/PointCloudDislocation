"""
Convenience functions for plotting point cloud features
"""

import numpy as np
import pandas as pd

from copy import copy

from itertools import combinations
from skimage.color import lab2rgb

# from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt

from utils import calculate_displacements_profile, transform_displacements


def initialize_plot_settings():
    plt.style.use("ggplot")
    matplotlib.rcParams["figure.figsize"] = (8, 8)
    matplotlib.rcParams["font.size"] = 14
    matplotlib.rcParams["axes.labelsize"] = 18
    matplotlib.rcParams["xtick.labelsize"] = 14
    matplotlib.rcParams["ytick.labelsize"] = 14
    matplotlib.rcParams["legend.title_fontsize"] = 14
    matplotlib.rcParams["legend.fontsize"] = 14

    matplotlib.rcParams["axes.labelcolor"] = "k"

    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.grid"] = False

    matplotlib.rcParams["axes.facecolor"] = "w"
    matplotlib.rcParams["axes.edgecolor"] = "k"
    matplotlib.rcParams["xtick.color"] = "k"
    matplotlib.rcParams["ytick.color"] = "k"

    matplotlib.rcParams["legend.frameon"] = False

    matplotlib.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["savefig.pad_inches"] = 0

def plot_residuals(ux, uy, uz, uxtrue, uytrue, uztrue, **kwargs):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    im = ax[0].imshow(
        uxtrue - ux,
        cmap="RdBu_r",
        origin="lower",
        **kwargs,
    )
    plt.colorbar(
        im,
        label="$\Delta u_x$ [m]",
        ax=ax[0],
        orientation="horizontal",
        extend="both",
        shrink=0.75,
    )
    ax[0].set_axis_off()

    im = ax[1].imshow(
        uytrue - uy,
        cmap="RdBu_r",
        origin="lower",
        **kwargs,
    )
    plt.colorbar(
        im,
        label="$\Delta u_y$ [m]",
        ax=ax[1],
        orientation="horizontal",
        extend="both",
        shrink=0.75,
    )
    ax[1].set_axis_off()

    im = ax[2].imshow(
        uztrue - uz,
        cmap="RdBu_r",
        origin="lower",
        **kwargs,
    )
    plt.colorbar(
        im,
        label="$\Delta u_z$ [m]",
        ax=ax[2],
        orientation="horizontal",
        extend="both",
        shrink=0.75,
    )
    ax[2].set_axis_off()

def plot_results(
    x,
    y,
    ux,
    uy,
    uz,
    depth=None,
    ds=None,
    strike=None,
    ax=None,
    letters=["A", "B", "C"],
    remove_first_axes=True,
    plot_colorbars=False,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    im = ax[0].imshow(
        ux,
        cmap="RdBu_r",
        origin="lower",
        extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
        **kwargs,
    )
    if plot_colorbars:
        plt.colorbar(
            im,
            label="Displacement $u_x$ [m]",
            ax=ax[0],
            orientation="horizontal",
            extend="both",
            shrink=0.75,
            pad=0.25,
        )
    if remove_first_axes:
        ax[0].set_axis_off()
    else:
        ax[0].set_xticks([321000, 322000])
        ax[0].set_yticks([4164500, 4165250])
        ax[0].set_xlabel("Easting [m]")
        ax[0].set_ylabel("Northing [m]")

    im = ax[1].imshow(
        uy,
        cmap="RdBu_r",
        origin="lower",
        extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
        **kwargs,
    )
    if plot_colorbars:
        plt.colorbar(
            im,
            label="Displacement $u_y$ [m]",
            ax=ax[1],
            orientation="horizontal",
            extend="both",
            shrink=0.75,
            pad=0.25,
        )
    ax[1].set_axis_off()

    im = ax[2].imshow(
        uz,
        cmap="RdBu_r",
        origin="lower",
        extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
        **kwargs,
    )
    if plot_colorbars:
        plt.colorbar(
            im,
            label="Displacement $u_z$ [m]",
            ax=ax[2],
            orientation="horizontal",
            extend="both",
            shrink=0.75,
            pad=0.25,
        )
    ax[2].set_axis_off()

    for axis, letter in zip(ax, letters):
        axis.text(0.025, 0.9, letter, transform=axis.transAxes, fontsize=14)


"""
    plt.suptitle(
        f"Displacements: dip slip = {ds:.2f} m, depth = {depth:.0f} m, strike = {strike:.0f} deg, dip = 90 deg",
        fontsize=16,
    )
"""


def plot_results_twopanel(x, y, ux, uy, uz, umax=5, **kwargs):
    xx, yy = np.meshgrid(x, y)
    uh = np.sqrt(ux ** 2 + uy ** 2)
    mask = np.abs(uh) < umax
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].quiver(
        xx[mask],
        yy[mask],
        ux[mask],
        uy[mask],
        units="xy",
        angles="xy",
        scale_units="xy",
        headlength=15,
        headwidth=10,
    )
    ax[0].set_axis_off()
    ax[0].axis("square")
    ax[0].text(0.025, 0.9, "A", transform=ax[0].transAxes, fontsize=18)
    im = ax[1].imshow(uz, cmap="RdBu_r", origin="lower", **kwargs)
    ax[1].set_axis_off()
    ax[1].axis("square")
    ax[1].text(0.025, 0.9, "B", transform=ax[1].transAxes, fontsize=18)


def plot_displacement_profile(
    us,
    labels,
    ylabel="Displacement $u_z$ [m]",
    legend_title="Dip slip",
    dx=25,
    dip_slip=0,
    strike_slip=0,
    plot_model=True,
    xlim=None,
    ylim=None,
):
    ij = []
    j = 0
    for i in np.arange(0, us[0].shape[0], dtype=int):
        ij.append([i, j])
        j += 1
        if j >= us[0].shape[1]:
            break
    ij = np.array(ij)

    plt.figure(figsize=(10, 5))

    markers = ["o", "s", "^"]
    for u, m, label in zip(us, markers, labels):
        plt.plot(
            u[ij[:, 0], ij[:, 1]],
            m,
            ms=6.5,
            mfc="gray",
            mec="k",
            ls="None",
            label=label,
        )

    plt.xticks([10, 20, 30, 40])
    xt = np.array(plt.xticks()[0])
    plt.gca().set_xticklabels(dx * xt)
    ymax = np.nanmax(u[ij[:, 0], ij[:, 1]])
    plt.ylim([-1.05 * ymax, 1.05 * ymax])

    plt.xlabel("Distance [m]")
    plt.ylabel(ylabel)
    plt.legend(title=legend_title)
    plt.gcf().patch.set_facecolor("w")


def plot_displacement_profile_resolution(
    us,
    labels,
    markers,
    ylabel="Displacement $u_z$ [m]",
    dxs=[5, 12.5, 25, 50],
    strike=0,
    depth=100,
    dip_slip=0,
    strike_slip=0,
    plot_model=False,
    xlim=None,
    ylim=None,
):
    ys = []
    ymods = []
    rmses = []
    if plot_model:
        ux, uy, uz = calculate_displacements_profile(
            strike=310,
            depth=depth,
            dip_slip=dip_slip,
            strike_slip=strike_slip,
            xlim=xlim,
            ylim=ylim,
        )
        ij = []
        j = 0
        for i in np.arange(0, ux.shape[0], dtype=int):
            ij.append([i, j])
            j += 1
            if j > ux.shape[1]:
                break
        ij = np.array(ij)

        xm = np.arange(ij.shape[0])
        if dip_slip > 0:
            um = uz[ij[:, 0], ij[:, 1]]
        if strike_slip > 0:
            ux = ux[ij[:, 0], ij[:, 1]]
            uy = uy[ij[:, 0], ij[:, 1]]
            um = transform_displacements(ux, uy, theta=strike)

    fig, axes = plt.subplots(len(us), 1, figsize=(6, 3 * len(us)), sharex=True)
    letters = ["F", "G", "H", "I", "J"]
    for u, ax, m, dx, label, letter in zip(us, axes, markers, dxs, labels, letters):
        ij = []
        j = 0
        for i in np.arange(0, u.shape[0], dtype=int):
            ij.append([i, j])
            j += 1
            if j >= u.shape[1]:
                break
        ij = np.array(ij)

        xp = dx * np.arange(ij.shape[0])
        up = u[ij[:, 0], ij[:, 1]]
        mask = np.isfinite(up)
        ax.plot(xp[mask], up[mask], m, mfc="w", mec="k", ls="None")

        if plot_model:
            ax.plot(xm, um, "k-")
            idx = list(map(int, xp[mask]))
            ys.append(up[mask])
            ymods.append(um[idx])
            rmses.append(np.sqrt(np.nanmean((um[idx] - up[mask]) ** 2)))

        ax.text(0.025, 0.9, letter, transform=ax.transAxes, fontsize=14)
        ax.text(0.75, 0.1, f"$dx$ = {dx:.1f} m", transform=ax.transAxes, fontsize=14)
        ax.set_xlim([150, 1100])
        ymax = np.max(up[mask])
        ax.set_ylim([-0.65, 0.65])
        ax.set_yticks([-0.5, 0, 0.5])

    axes[-1].set_xticks([200, 650, 1050])
    axes[-1].set_xticklabels(["0", "450", "850"])
    axes[-1].set_xlabel("Distance [m]")
    axes[2].set_ylabel(ylabel)

    return ys, ymods, rmses


def plot_points_perspective(data, ax=None, **kwargs):
    x = data[:, 0]
    y = data[:, 1]
    colors = copy(data[:, 3:6])
    colors /= colors.max()

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.scatter(x, y, marker=".", c=colors, s=2, **kwargs)

    ax.set_aspect("equal")
