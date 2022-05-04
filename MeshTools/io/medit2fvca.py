# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 09:17:57 2015

@author: lopez
"""

import os
import numpy as np
from emerge.io.iomedit import MeditInfo
import emerge.io.iofvca as fvca
import vtkwriters as vtkw

fcva_writer = {
    "standard": fvca.write,
    "compass": fvca.write_compass_format,
    #               'compass-appendnodes':fvca.write_compass_appnode_format
}

filename = "out.mesh"
# filename = 'foo.mesh'
basename = os.path.splitext(filename)[0]

mesh = MeditInfo(filename)
mesh.compute_info()
mesh.output_vtu(basename)

# look for fault indexes
indexfile = filename.replace(".mesh", ".indexes")
fracture_indexes = list()
top_index = None
bottom_index = None
xmin_index = None
xmax_index = None
if os.path.exists(indexfile):
    with open(indexfile) as f:
        for line in f:
            l = line.split()
            if l[1].strip().lower().startswith("fault"):
                fracture_indexes.append(int(l[0]))
            if l[1].strip().lower().startswith("boundary"):
                if l[-1].strip().lower() == "xmin":
                    assert xmin_index is None
                    xmin_index = int(l[0])
                if l[-1].strip().lower() == "zmin":
                    assert bottom_index is None
                    bottom_index = int(l[0])
                if l[-1].strip().lower() == "xmax":
                    assert xmax_index is None
                    xmax_index = int(l[0])
            if l[1].strip().lower().startswith("topography"):
                assert top_index is None
                top_index = int(l[0])
# print fracture_indexes
# print dirichlet_indexes

# select faces to export
tagged_faces = mesh.faces_indexes(mesh.triangles)
fracture_faces = tagged_faces[np.in1d(mesh.triangles_index, fracture_indexes)]
bottom_faces = tagged_faces[np.in1d(mesh.triangles_index, [bottom_index])]
top_faces = tagged_faces[np.in1d(mesh.triangles_index, [top_index])]
x_faces = tagged_faces[np.in1d(mesh.triangles_index, [xmin_index, xmax_index])]
# Locate aquifer boundary faces
boundary_faces = np.where(mesh.facecells[:, 1] < 0)[0]
boundary_cells = mesh.facecells[boundary_faces, 0]
aquifer_boundary_faces = boundary_faces[
    np.where(mesh.cells_index[boundary_cells] == 2)[0]
]
assert np.all(mesh.cells_index[mesh.facecells[aquifer_boundary_faces, 0]] == 2)
xaquifer_faces = np.intersect1d(x_faces, aquifer_boundary_faces, assume_unique=True)
dirichlet_faces = np.union1d(top_faces, bottom_faces)
dirichlet_faces = np.union1d(dirichlet_faces, xaquifer_faces)
faces_index = np.zeros(mesh.faces.shape[0], dtype="i")
faces_index[fracture_faces] = -2
faces_index[dirichlet_faces] = 1
# Modify top layer rocktype
mesh.cells_index[mesh.cells_index == 3] = 1

fcva_writer["compass"](basename, mesh, faces_index)

# Append node information
dirichlet_nodes = np.unique(np.ravel(mesh.faces[dirichlet_faces]))
IdNode = np.zeros(mesh.vertices.shape[0], dtype=np.int)
IdNode[dirichlet_nodes] = 1
with open(basename + ".msh", "a") as f:
    f.write("Node info: 1 for dirichlet node\n")
    np.savetxt(f, IdNode, fmt="%d")

# boundary fracture nodes
bottom_nodes = np.unique(np.ravel(mesh.faces[bottom_faces]))
fracture_nodes = np.unique(np.ravel(mesh.faces[fracture_faces]))
bfn = bottom_nodes[np.in1d(bottom_nodes, fracture_nodes, assume_unique=True)]
C0 = np.zeros(mesh.vertices.shape[0], dtype=np.double)
C0[bfn] = 1.0
np.savetxt("initial_concentration.dat", C0)

# pressure
Pin = 1.0
Pout = 0.0
pressure = Pin + (Pout - Pin) * mesh.vertices[:, 2]
np.savetxt("pressure_gradient.dat", pressure)

vtkw.write_vtu(
    vtkw.vtu_doc(
        mesh.vertices,
        mesh.cells,
        pointdata={"IdNode": IdNode, "C0": C0},
        celldata={"material": mesh.cells_index},
    ),
    "check_boundary_conditions",
)

# import shutil
# shutil.copy('out.msh', 'd:/work/devel/compass/Mesh/out.msh')
# shutil.copy('pressure_gradient.dat', 'd:/work/devel/compass/Mesh/P0.dat')
# shutil.copy('initial_concentration.dat', 'd:/work/devel/compass/Mesh/C0.dat')
