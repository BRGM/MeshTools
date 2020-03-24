import sys
import os
import numpy as np
import MeshTools.io.petrel as petrel
import MeshTools.vtkwriters as vtkw

filename = sys.argv[1]
mesh, perm = petrel.import_eclipse_grid(filename)
offsets, cellsnodes = mesh.cells_nodes_as_COC()
vtkw.write_vtu(
    vtkw.vtu_doc_from_COC(
        mesh.vertices_array(),
        np.array(offsets[1:], copy=False),  # no first zero offset for vtk
        np.array(cellsnodes, copy=False),
        mesh.cells_vtk_ids(),
        celldata={
            name: np.asarray(values) for name, values in zip(["kx", "ky", "kz"], perm)
        },
    ),
    os.path.splitext(filename)[0] + ".vtu",
)
