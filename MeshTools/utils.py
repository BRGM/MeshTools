# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:48:35 2018

@author: lopez
"""

import numpy as np
from scipy.interpolate import interp1d

import MeshTools as MT
import MeshTools.GridTools as GT

def make_saw(pinch):
    return interp1d(
        [0., 1/3, 2/3, 1.],
        [0.5, pinch, 1. - pinch, 0.5],
        bounds_error=True
    )

def kershaw_mesh(shape, pinchx=0.1, pinchy=None, output_ijk=False):
    nx, ny, nz = shape
    grid = GT.grid2hexs(shape=(nx, ny, nz), output_ijk=output_ijk)
    vertices, cells = grid[:2]
    z = vertices[:, 2]
    under = z < 0.5
    above = np.logical_not(under)
    def saw_axis(axis, pinch):
        saw = make_saw(pinch)(vertices[:, axis])
        z[under]*= saw[under] / 0.5
        z[above] = 1. - (1. - z[above]) * (1. - saw[above]) / 0.5
    if pinchx:
        saw_axis(0, pinchx)
    if pinchy:
        saw_axis(1, pinchy)
    mesh = MT.HexMesh.make(vertices, MT.idarray(cells))
    if output_ijk:
        return mesh, grid[-1]
    return mesh
