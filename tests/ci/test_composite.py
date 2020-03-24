import numpy as np

from MeshTools import vtkwriters as vtkw

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
diamond_faces = [
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 1],
    [4, 1, 2],
    [4, 2, 3],
    [4, 3, 1]
]
tet_faces = [
    [2, 3, 4],
    [2, 3, 5],
    [3, 4, 5],
    [4, 2, 5],
]
# fmt: on


def test_write_composite():
    vtkw.write_vtu(vtkw.polyhedra_vtu_doc(vertices, [diamond_faces]), "diamond")
    vtkw.write_vtu(vtkw.polyhedra_vtu_doc(vertices, [tet_faces]), "tet")
    vtkw.write_vtm(vtkw.vtm_doc(["diamond.vtu", "tet.vtu"]), "composite")
    vtkw.write_vtm(
        vtkw.vtm_doc(
            {"copy1": ["diamond.vtu", "tet.vtu"], "copy2": ["diamond.vtu", "tet.vtu"],}
        ),
        "composite_with_duplicates",
    )


if __name__ == "__main__":
    test_write_composite()
