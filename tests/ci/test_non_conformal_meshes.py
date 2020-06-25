import MeshTools.io.dummy_petrel_grids as dummy_grids
from test_dummy_petrel_grids import build_and_dump

regular = dummy_grids.build_regular(4, 2, xres=5, yres=5, zres=10)
regular = dummy_grids.refine(regular, 2)
regular = regular[:, None, ...]
build_and_dump("regular", regular)

mesh = dummy_grids.build_one_unconformity_mesh(4, 2, 3, xres=5, yres=5, zres=10)
mesh = dummy_grids.refine(mesh, 2)
mesh = mesh[:, None, :, ...]
build_and_dump("one_unconformity_mesh", mesh)

mesh2 = dummy_grids.build_non_conformal_mesh(4, 2, 2, xres=5, yres=5, zres=10)
mesh2 = dummy_grids.refine(mesh2, 2)
mesh2 = mesh2[:, None, :, ...]
build_and_dump("non_conformal_mesh", mesh2)
