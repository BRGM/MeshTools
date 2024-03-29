# from collections import namedtuple
import numpy as np

import vtkwriters as vtkw

# from MeshTools.io.petrel import PetrelGrid
# from MeshTools.RawMesh import RawMesh
# import MeshTools.io.dummy_petrel_grids as dummy_grids

# import pytest


# fmt: off
vertices = np.array(
    [
        [ 0,  0, -1],
        [ 0,  0,  0],
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1],
        [ 1,  1,  1],
    ],
    dtype=np.float64,
)
faces = [
    [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [4, 1, 2],
        [4, 2, 3],
        [4, 3, 1]
    ],
    [
        [2, 3, 4],
        [2, 3, 5],
        [3, 4, 5],
        [4, 2, 5],
    ],
]
# fmt: on


def test_write_dummy_polyhedra():
    vtkw.write_vtu(vtkw.polyhedra_vtu_doc(vertices, faces), "diamond_and_test")
