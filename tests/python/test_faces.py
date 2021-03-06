import numpy as np
import MeshTools as MT


def test_faces():

    Point = MT.Point
    Tet = MT.Tetrahedron
    Wedge = MT.Wedge

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

    mesh = MT.HybridMesh.Mesh()
    vertices = mesh.vertices
    for P in pts + pts2:
        vertices.append(P)

    cellnodes = mesh.connectivity.cells.nodes
    for elt in tets + wedges:
        cellnodes.append(elt)

    mesh.connectivity.update_from_cellnodes()

    faces = mesh.connectivity.faces
    print(faces)
    face_cells = faces.cells
    a = np.asarray(face_cells)
    for pair in a:
        print(pair)

    a = faces.cells_as_array()
    print(a)


if __name__ == "__main__":
    test_faces()
