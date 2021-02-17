import numpy as np
import MeshTools


def make_mesh():

    Point = MeshTools.Point
    Tet = MeshTools.Tetrahedron
    Wedge = MeshTools.Wedge

    pts = [
        Point((0.0, 0.0, -1.0)),
        Point((1.0, -1.0, 0.0)),
        Point((1.0, 1.0, 0.0)),
        Point((-1.0, 0.0, 0.0)),
        Point((0.0, 0.0, 1.0)),
    ]

    pts2 = [Point((2.0, 0.0, -1.0)), Point((2.0, -1.0, 0.0)), Point((2.0, 1.0, 0.0))]

    tets = [Tet((1, 2, 3, 0)), Tet((1, 2, 3, 4))]

    wedges = [Wedge((0, 1, 2, 5, 6, 7))]

    mesh = MeshTools.HybridMesh.Mesh()
    vertices = mesh.vertices
    for P in pts + pts2:
        vertices.append(P)

    cellnodes = mesh.connectivity.cells.nodes
    for elt in tets + wedges:
        cellnodes.append(elt)

    mesh.connectivity.update_from_cellnodes()

    return mesh


def test_hybridmesh():
    mesh = make_mesh()
    MeshTools.to_vtu(mesh, "mesh-test3.vtu")
    assert mesh.nb_cells == 3


def test_findfaces():
    from MeshTools.utils import find_faces

    mesh = make_mesh()

    faces = [[0, 3, 1], [1, 2, 7, 6]]
    found, notfound = find_faces(mesh, faces)
    assert tuple(np.unique(found)) == (1, 9)
    assert notfound is None

    faces = [[7, 5, 4], [0, 3, 1], [1, 2, 7, 6], [1, 2]]
    found, notfound = find_faces(mesh, faces)
    assert tuple(np.unique(found)) == (1, 9)
    assert tuple(np.unique(notfound)) == (0, 3)
