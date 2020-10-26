import numpy as np
import MeshTools as MT


def test_wedge():
    vertices = np.array(
        [(0, 0, 0), (1, 1, 0), (0, 2, 0), (0, 0, 1), (1, 1, 1), (0, 2, 1)], dtype="d"
    )
    wedge = np.array([(0, 1, 2, 3, 4, 5)])
    mesh = MT.WedgeMesh.make(vertices, wedge)
    MT.to_vtu(mesh, "wedge.vtu")


if __name__ == "__main__":
    test_wedge()
