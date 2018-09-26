# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage.measure import marching_cubes_lewiner as marching_cubes
import MeshTools as MT
import MeshTools.CGALWrappers as CGAL
import MeshTools.vtkwriters as vtkw

shape = nx, ny, nz = (50,)*3
steps = (
    np.linspace(0, 1, nx),
    np.linspace(0, 1, ny),
    np.linspace(0, 1, nz),
)
coordinates = np.meshgrid(*steps, indexing='ij')
points = np.stack(coordinates, axis=-1)
points.shape = (-1, 3)

def f(pts):
    return pts[:,2] + 0.5 * np.cos(pts[:, 0]*2*np.pi)*np.sin(pts[:, 1]*2*np.pi)

field_value = f(points)
field_value.shape = shape
isocontour = marching_cubes(field_value)

def rescale_into_unit_box(pts):
    return pts * np.array([
        1 / (nx - 1), 1 / (ny - 1), 1 / (nz - 1)
    ])

as_tsurf = lambda t: MT.TSurf.make(t[0], MT.idarray(t[1]))

verts, faces, normals, values = isocontour
verts = rescale_into_unit_box(verts)
MT.to_vtu(
    as_tsurf((verts, faces)),
    '%s_eggs.vtu' % (os.path.splitext(__file__)[0]),
)

tsurf = CGAL.TSurf(verts, faces)
offseted_tsurf = CGAL.TSurf(verts + np.array([0.3, 0.2, 0.1]), faces)

# clip surface
fc = offseted_tsurf.face_centers()
# line: x + y - 1 = 0
offseted_tsurf.remove_faces(fc[:, 0] + fc[:, 1] - 1 < 0)

verts, faces = offseted_tsurf.as_arrays() 
MT.to_vtu(
    as_tsurf((verts, faces)),
    '%s_offseted_eggs.vtu' % (os.path.splitext(__file__)[0]),
)

polylines = CGAL.intersection_curves(tsurf, offseted_tsurf)

for pi, polyline in enumerate(polylines):
    vtkw.write_vtu(
        vtkw.polyline_as_vtu(np.array(polyline)),
        'polyline_%03d' % pi
    )

