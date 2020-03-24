from pathlib import Path

import numpy as np
import MeshTools


def extract(filename, cols=None, dtype=int, sep=" "):
    filepath = Path(filename)
    assert filepath.exists()
    res = []
    if cols is None:
        append_line = lambda l: res.append([dtype(s) for s in l])
    elif type(cols) is slice:
        append_line = lambda l: res.append([dtype(s) for s in l[cols]])
    else:
        append_line = lambda l: res.append([dtype(l[k]) for k in cols])
    with filepath.open() as f:
        for line in f:
            l = line.strip().split(sep)
            append_line(l)
    return res


vertices = extract("NODES.txt", dtype=float, cols=slice(1, None))
cells = extract("MIXED.txt", cols=slice(1, None))

vertices = np.array(vertices)
# Zero based indexing
cells = [MeshTools.idarray(cell) - 1 for cell in cells]

# convert cells to elements
elements = []
for cell in cells:
    if len(cell) == 6:
        elements.append(MeshTools.Wedge(cell))
    else:
        assert len(cell) == 8, "Cell should have 6 (wedge) or 8 (hexaedron) vertices"
        elements.append(MeshTools.Hexahedron(cell))

mesh = MeshTools.HybridMesh.create(vertices, elements)

MeshTools.to_vtu(mesh, "my_hybrid_mesh")
