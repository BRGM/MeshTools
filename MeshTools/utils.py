# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:48:35 2018

@author: lopez
"""

import numpy as np

from ._MeshTools import idtype
from . import GridTools as GT
from . import vtkwriters as vtkw
from . import RawMesh


def idarray(a):
    return np.asarray(a, dtype=idtype())


def to_vtu(mesh, filename, **kwargs):
    if type(mesh) is RawMesh:
        cell_faces = mesh.cell_faces
        face_nodes = mesh.face_nodes
        vtu = vtkw.polyhedra_vtu_doc(
            mesh.vertices,
            [[face_nodes[face] for face in faces] for faces in cell_faces],
            **kwargs
        )
    else:
        offsets, cellsnodes = mesh.cells_nodes_as_COC()
        vtu = vtkw.vtu_doc_from_COC(
            mesh.vertices_array(),
            np.array(offsets[1:], copy=False),  # vtk: no first zero offset
            np.array(cellsnodes, copy=False),
            mesh.cells_vtk_ids(),
            **kwargs
        )
    vtkw.write_vtu(vtu, filename)


def grid3D(**kwargs):
    if "steps" in kwargs:
        assert len(kwargs) == 1
        vertices, hexs = GT.steps2hex(kwargs["steps"], idtype=idtype())
    else:
        assert not "steps" in kwargs
        vertices, hexs = GT.grid2hexs(**kwargs, idtype=idtype())
    return HexMesh.make(vertices, hexs)


# def grid3D(**kwargs):
#    vertices, hexs = GT.grid2hexs(**kwargs, idtype=idtype())
#    return HexMesh.make(vertices, hexs)

## Tet Volumes
# A = mesh.vertices[mesh.cellnodes[:, 0]]
# AB, AC, AD = (mesh.vertices[mesh.cellnodes[:, i+1]] - A for i in range(3))
# det = AB[:, 0] * AC[:, 1] * AD[:, 2]
# det+= AB[:, 1] * AC[:, 2] * AD[:, 0]
# det+= AB[:, 2] * AC[:, 0] * AD[:, 1]
# det-= AB[:, 2] * AC[:, 1] * AD[:, 0]
# det-= AB[:, 0] * AC[:, 2] * AD[:, 1]
# det-= AB[:, 1] * AC[:, 0] * AD[:, 2]
# vol = np.abs(det) / 6.

# print(vol.min(), vol.max())
