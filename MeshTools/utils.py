# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:48:35 2018

@author: lopez
"""

import numpy as np

from ._MeshTools import idtype, HexMesh
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


def extrude(vertices, polygons, offsets):
    """
    Extrude a polygon mesh according successing offsets.
    :param vertices: the 3D vertices of the surface mesh
    :param polygons: the connectivity table of the polygons (all polygons must be of the same type)
    :param offsets: a sequence of 3D vector presenting the succesive offsets
    :return: a tuple of vertices and 3D cell connectivity
    """
    vertices = np.asarray(vertices, dtype=np.double)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    offsets = np.asarray(offsets, dtype=np.double)
    assert offsets.ndim == 2 and offsets.shape[1] == 3
    polygons = np.asarray(polygons, dtype=np.double)
    cum_offsets = np.cumsum(offsets, axis=0)
    new_vertices = np.vstack([vertices,] + [vertices + v for v in cum_offsets])
    nv = vertices.shape[0]
    layer = np.hstack([polygons, polygons + nv])
    nl = offsets.shape[0]
    cells = np.vstack([layer + k * nv for k in range(nl)])
    return new_vertices, idarray(cells)


def axis_extrusion(vertices, polygons, offsets, axis=-1):
    """
    Extrude a mesh along a given axis.
    Extrude a polygon mesh according successing offsets.
    :param vertices: the 3D vertices of the surface mesh
    :param polygons: the connectivity table of the polygons (all polygons must be of the same type)
    :param offsets: a sequence of length presenting the succesive offsets along axis `axis`
    :param axis: the extrusion axis (defaults to the latest axis)
    :return: a tuple of vertices and 3D cell connectivity
    """
    offsets = np.asarray(offsets, dtype=np.double)
    assert offsets.ndim == 1
    offsets_3d = np.zeros((offsets.shape[0], 3), dtype=np.double)
    offsets_3d[:, axis] = offsets
    return extrude(vertices, polygons, offsets_3d)


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


def find_faces(mesh, faces, fortran_indexing=False):
    """
    :param mesh: a MeshTool mesh (typically an hybrid mesh)
    :param faces: a (possibly heterogeneous) list of faces as list of nodes
    :param fortran_indexing: True if node numbering start at 1
    """

    pointers, nodes = mesh.faces_nodes_as_COC()
    pointers = pointers.array_view()
    nodes = nodes.array_view()
    facesizes = pointers[1:] - pointers[:-1]
    sizes = np.unique(facesizes)

    elements = {}
    for size in sizes:
        elts = (facesizes == size).nonzero()[0]
        elt_nodes = np.array(
            [nodes[int(p) : (int(p) + int(size))] for p in pointers[elts]]
        )
        elt_nodes.sort(axis=1)
        elements[size] = (elts, elt_nodes, np.dtype([("", nodes.dtype)] * size))

    fsizes = np.array([len(f) for f in faces])

    felements = {}
    for size in np.unique(fsizes):
        elts = (fsizes == size).nonzero()[0]
        elt_nodes = np.array([faces[k] for k in elts], dtype=nodes.dtype)
        elt_nodes.sort(axis=1)
        if fortran_indexing:
            elt_nodes -= 1
        felements[size] = (elts, elt_nodes)

    found = []
    notfound = []

    for size, (elts, nodes) in felements.items():
        if size in elements:
            melts, mnodes, dt = elements[size]
            _, foo, indices = np.intersect1d(
                nodes.view(dt), mnodes.view(dt), return_indices=True
            )
            found.append(melts[indices])
            assert len(foo) <= len(elts)
            if len(foo) < len(elts):
                missed = np.ones(len(elts), dtype=np.bool)
                missed[foo] = False
                notfound.append(elts[missed.nonzero()[0]])
        else:
            notfound.append(elts)

    pack = lambda l: np.hstack(l) if len(l) > 0 else None
    return pack(found), pack(notfound)
