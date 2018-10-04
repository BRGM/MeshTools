import numpy as np
import MeshTools as MT
from MeshTools.RawMesh import RawMesh

vertices = np.array([
    (0, 0, 0), # -- cube vertices
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (2, 0.5, 0.5) # <- pyramid tip
], dtype=np.double)

face_nodes = [
    (0, 1, 2, 3), # -- cube faces
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5), # <- face shared with pyramid
    (2, 3, 7, 6),
    (3, 0, 4, 7),
    (1, 2, 8), # -- pyramid triangles
    (2, 6, 8),
    (6, 5, 8),
    (5, 1, 8),
]

cell_faces = [
    (0, 1, 2, 3, 4, 5), # cube
    (3, 6, 7, 8, 9), # pyramid
]

def test_raw_mesh():
    mesh = RawMesh(
        vertices=vertices,
        face_nodes=face_nodes,
        cell_faces=cell_faces
    )
    cell_property = np.array([np.sqrt(2), np.pi]) # 2 cells
    tetmesh, original_cell = mesh.as_tets()
    MT.to_vtu(
        tetmesh, 'cells_as_tets',
        celldata = {
            'original_cell': original_cell,
            'magic_numbers': cell_property[original_cell],
        },
    )


if __name__=='__main__':
    test_raw_mesh()
