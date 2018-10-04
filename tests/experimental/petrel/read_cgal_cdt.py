# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:21:11 2018

@author: lopez
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

filename = 'precc_triangulation.txt'

with open(filename) as f:
    nv, nf, _ = [int(s) for s in f.readline().strip().split()]
    vertices = np.array([
        [float(s) for s in f.readline().strip().split()]
        for _ in range(nv-1)        
    ])
    f.readline() # blank line
    faces = np.array([
        [int(s) for s in f.readline().strip().split()]
        for _ in range(nf)        
    ])
    f.readline() # blank line
    for _ in range(nf):
        f.readline()
    constraints = np.array([
        [True if s=='C' else False for s in f.readline().strip().split()]
        for _ in range(nf)        
    ])

finite_faces = faces[np.all(faces>0, axis=1)]
finite_constraints = constraints[np.all(faces>0, axis=1)]
finite_faces-= 1

ax = plt.gca()
ax.set_aspect('equal')
ax.triplot(vertices[:, 0], vertices[:, 1], finite_faces)

for fi, constraint in enumerate(finite_constraints):
    for vi, active in enumerate(constraint):
        if active:
            S = np.vstack([
                    vertices[finite_faces[fi, (vi+1)%3]],
                    vertices[finite_faces[fi, (vi+2)%3]]
                    ])
            plt.plot(S[:, 0], S[:, 1], 'r')

plt.savefig(os.path.splitext(filename)[0] + '.pdf')