from collections import namedtuple
import numpy as np

from MeshTools import to_vtu, HexMesh
from MeshTools.io.petrel import PetrelGrid
from MeshTools.RawMesh import RawMesh
import MeshTools.io.dummy_petrel_grids as dummy_grids

import pytest


def build_and_dump(name, hexaedra):
    print("-" * 80)
    print("processing", name)
    pil = dummy_grids.pilars(hexaedra)
    grid = PetrelGrid.build_from_arrays(pil[..., :3], pil[..., 3:], hexaedra[..., 2])
    print("pilars shape", pil.shape)
    print("zcorn shape", hexaedra[..., 2].shape)
    hexa, vertices, cell_faces, face_nodes = grid.process()
    print("minimum distance", dummy_grids.minimum_distance(vertices))
    to_vtu(HexMesh.make(dummy_grids.depth_to_elevation(vertices), hexa), name)
    mesh = RawMesh(vertices=vertices, face_nodes=face_nodes, cell_faces=cell_faces)
    print(
        "Original {} mesh with: {:d} vertices, {:d} hexaedra, {:d} faces".format(
            name, mesh.nb_vertices, mesh.nb_cells, mesh.nb_faces
        )
    )
    vertices, cell_faces, face_nodes = grid.process_faults(hexa)
    mesh = RawMesh(vertices=vertices, face_nodes=face_nodes, cell_faces=cell_faces)
    raw_mesh, original_cell = mesh.as_hybrid_mesh()
    print(
        "Splitted {} mesh with: {:d} vertices, {:d} cells, {:d} faces".format(
            name, raw_mesh.nb_vertices, raw_mesh.nb_cells, raw_mesh.nb_faces
        )
    )
    to_vtu(raw_mesh, name + "_splitted", celldata={"original_cell": original_cell})


DummyGrid = namedtuple("DummyGrid", ["name", "grid"])


@pytest.fixture(
    params=[
        DummyGrid("common_node", dummy_grids.common_node()),
        DummyGrid("sugar_box", dummy_grids.grid_of_heaxaedra((4, 3, 2))),
        DummyGrid("stairs", dummy_grids.four_cells_stairs()),
        DummyGrid("ramp", dummy_grids.faulted_ramp((8, 2, 1), begin=0.33)),
    ],
    ids=lambda dg: dg.name,
)
def dummy_grid(request):
    return request.param


def test_grid(dummy_grid):
    build_and_dump(*dummy_grid)


def test_various_dirty_checks_to_be_cleaned():
    dummy_grids.various_dirty_checks_to_be_cleaned()
