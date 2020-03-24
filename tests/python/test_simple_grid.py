# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 09:50:20 2018

@author: lopez
"""

import MeshTools as MT

n = 20
L = 1.0
mesh = MT.grid3D(shape=(n,) * 3, extent=(2 * L,) * 3, origin=(-L,) * 3)
MT.to_vtu(mesh, "potgrid.vtu")
