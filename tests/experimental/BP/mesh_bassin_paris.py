#
# This file is part of ComPASS.
#
# ComPASS is free software: you can redistribute it and/or modify it under both the terms
# of the GNU General Public License version 3 (https://www.gnu.org/licenses/gpl.html),
# and the CeCILL License Agreement version 2.1 (http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html).
#
import numpy as np
import MeshTools as MT
import pickle
from bassin_paris import misc
import vtkwriters as vtkw

#  Extraction des sommets, connectivités, et numéros de couche
vertices, cells, cell_layer = misc.extract_mesh()
# Vérification de la cohérence du maillage et extraction des tailles de maille,
# des nombres de noeuds, des coordonnées, et des coins
# !! Sort une grille régulière de dx/dy minimum
(
    dxmin,
    dymin,
    nb_inoeuds,
    nb_jnoeuds,
    coord,
    bottom_left,
    upper_right,
) = misc.check_mesh(vertices, cells, cell_layer)
ix, jy, x, y, z = coord
diagonal = upper_right - bottom_left
Lx, Ly = diagonal
# Calcul des épaisseurs sur grille régulière
thicknesses = misc.compute_thickness_layers(
    vertices, cells, cell_layer, nb_inoeuds, nb_jnoeuds, coord, bottom_left
)  # =0 si empty = 1 si existe
# Calcul de l'altitudes des surfaces de chaque couche
#  !! Surfaces correspond aux noeuds, et pas au centre des mailles
id_layers = np.unique(cell_layer)
nb_layers = id_layers.shape[0]
surfaces_base = misc.compute_surfaces(
    nb_layers, Lx, dxmin, Ly, dymin, bottom_left, cells, cell_layer, x, y, z, "base"
)
surfaces_top = misc.compute_surfaces(
    nb_layers, Lx, dxmin, Ly, dymin, bottom_left, cells, cell_layer, x, y, z, "top"
)
surfaces = np.empty(
    (nb_layers + 1, int(1 + Lx / dxmin), int(1 + Ly / dymin)), dtype="d"
)
surfaces[:-1] = surfaces_top
surfaces[-1] = surfaces_base[-1]
# On prend le masque de surface_top et on l'applique à surfaces_base
for i in range(nb_layers - 1):
    mask = surfaces_top[i + 1] == 9999
    surfaces[i + 1][mask] = surfaces_base[i][mask]
nx, ny = surfaces.shape[1:]
lat, lon = np.meshgrid(
    bottom_left[0] + np.arange(nx) * dxmin,
    bottom_left[1] + np.arange(ny) * dymin,
    indexing="ij",
)
lat = np.repeat(lat[:, :, np.newaxis], surfaces.shape[0], axis=2)
lon = np.repeat(lon[:, :, np.newaxis], surfaces.shape[0], axis=2)
surfaces = np.transpose(surfaces, (1, 2, 0))
new_vertices = np.hstack([v.reshape((-1, 1)) for v in [lat, lon, surfaces]])
# On prend que les mailles non masquées
node_is_good = surfaces != 9999
surfaces_ma = np.round(np.ma.masked_values(surfaces, 9999.0), 2)
for layer in range(1, surfaces_ma.shape[2]):
    toto = surfaces_ma[:, :, layer - 1] - surfaces_ma[:, :, layer]
    ix, iy = np.ma.where(toto == 0)
    node_is_good[ix, iy, layer] = False
# Numérotation des mailles depuis 0. Les numéros des noeuds 9999. sont présents
#  mais ne servent à rien
node_number = np.cumsum(node_is_good)
node_number = node_number - 1  # to start numbering at 0
node_number.shape = surfaces.shape
# node_number a l'indice du noeud (on n'a pas compter les noeuds avec 9999)
cell_is_good = thicknesses != 0
cell_is_good = np.transpose(cell_is_good, (1, 2, 0))
# on cree le tableau de connectivité pour toutes les cellules (bonnes et mauvaises)
all_cells = np.hstack(
    [
        v.ravel().reshape((-1, 1))
        for v in [
            node_number[:-1, :-1, 1:],
            node_number[1:, :-1, 1:],
            node_number[:-1, 1:, 1:],
            node_number[1:, 1:, 1:],
            node_number[:-1, :-1, :-1],
            node_number[1:, :-1, :-1],
            node_number[:-1, 1:, :-1],
            node_number[1:, 1:, :-1],
        ]
    ]
)
a = np.reshape(
    all_cells, (surfaces.shape[0] - 1, surfaces.shape[1] - 1, surfaces.shape[2] - 1, 8)
)
# on selectionne les vertices
new_vertices = new_vertices[node_is_good.ravel()]
# nouvelles cellules
new_cells = a[cell_is_good]
new_cells = new_cells[:, [0, 1, 3, 2, 4, 5, 7, 6]]
new_cell_layer = id_layers[np.nonzero(cell_is_good)[2]]
# surfaces affleurantes
surf_affl = np.zeros(cell_is_good.shape[:-1])
for i in range(cell_is_good.shape[2]):
    couche = cell_is_good[:, :, i]
    surf_affl[couche] = np.where(surf_affl[couche] == 0, i + 1, surf_affl[couche])
new_surf_affl = np.zeros(cell_is_good.shape)
for i in range(cell_is_good.shape[2]):
    new_surf_affl[:, :, i] = np.where(
        surf_affl == i + 1, surf_affl, new_surf_affl[:, :, i]
    )
new_surf_affl = new_surf_affl[cell_is_good]
# surfaces bottoms
surf_bot = np.zeros(cell_is_good.shape[:-1])
for i in range(cell_is_good.shape[2] - 1, 0, -1):
    couche = cell_is_good[:, :, i]
    surf_bot[couche] = np.where(surf_bot[couche] == 0, i + 1, surf_bot[couche])
new_surf_bot = np.zeros(cell_is_good.shape)
for i in range(cell_is_good.shape[2] - 1, 0, -1):
    new_surf_bot[:, :, i] = np.where(surf_bot == i + 1, surf_bot, new_surf_bot[:, :, i])
new_surf_bot = new_surf_bot[cell_is_good]
del surf_bot, surf_affl
# colonnes
nb_cells_surface = cell_is_good.shape[0] * cell_is_good.shape[1]
new_cell_col = np.array(range(1, nb_cells_surface + 1))
new_cell_col = new_cell_col.reshape(cell_is_good.shape[:-1])
new_cell_col = np.repeat(new_cell_col[:, :, np.newaxis], cell_is_good.shape[2], axis=2)
new_cell_col = new_cell_col[cell_is_good]
#  On enlève les arrêtes nulles
arnul = misc.ArretesNulles(
    new_cell_col, new_vertices, new_cells, new_cell_layer, new_surf_affl, new_surf_bot
)
arnul.one_vertice()
print("one_vertice done")
arnul.two_vertices()
print("two_vertices done")
arnul.three_vertices()
print("three_vertices done")
arnul.set_cells()
typ = {
    "wedges": MT.Wedge,
    "pyramids": MT.Pyramid,
    "tetras": MT.Tetrahedron,
    "hexas": MT.Hexahedron,
}

paris = {}
paris["vertices"] = arnul.vertices
paris["cells_nodes"] = arnul.cells_nodes
paris["cells_faces"] = arnul.cells_faces
paris["faces_nodes"] = arnul.faces_nodes
paris["faces_nodes_top"] = arnul.faces_nodes_affl
paris["faces_nodes_bot"] = arnul.faces_nodes_bot
paris["cells_layers"] = arnul.cells_layers
paris["cells_top"] = arnul.cells_affls
paris["cells_bot"] = arnul.cells_bots

with open("bassin_paris/meshs.pkl", "bw") as fichier:
    pickle.dump(paris, fichier)

# Représentation sous vtk :
for poly in paris["cells_nodes"]:
    mesh = MT.HybridMesh.Mesh()
    vertices = mesh.vertices
    for P in paris["vertices"]:
        vertices.append(MT.Point(P))
    cellnodes = mesh.connectivity.cells.nodes
    for elt in paris["cells_nodes"][poly]:
        cellnodes.append(typ[poly](MT.idarray(elt)))
    mesh.connectivity.update_from_cellnodes()
    offsets, cellsnodes = mesh.cells_nodes_as_COC()
    vtkw.write_vtu(
        vtkw.vtu_doc_from_COC(
            mesh.vertices_array(),
            np.array(offsets[1:], copy=False),  # no first zero offset for vtk
            np.array(cellsnodes, copy=False),
            mesh.cells_vtk_ids(),
        ),
        "{0}.vtu".format(poly),
    )
#  python postprocess_snapshots -s output-test-bass_paris
