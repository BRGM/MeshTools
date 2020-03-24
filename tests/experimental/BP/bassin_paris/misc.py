# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 05:43:29 2017

@author: lopez
"""

import numpy as np
from scipy import interpolate

xyzfile = "bassin_paris/input/nohi_newep.xyz"
connectivityfile = "bassin_paris/input/nohi_newep.nsom"
layerfile = "bassin_paris/input/Numeros-de-couches"


def extract_mesh():
    cell_layer = np.loadtxt(layerfile, usecols=(1,), dtype="i")
    vertices = np.loadtxt(xyzfile)
    cellfilename = connectivityfile
    newcellfilename = cellfilename + ".new"
    # restack
    with open(cellfilename) as fin:
        with open(newcellfilename, "w") as fout:
            line = fin.readline()
            while line.strip():
                fout.write(line.strip())
                line = fin.readline()
                fout.write(line)
                line = fin.readline()
    cells = np.loadtxt(newcellfilename, usecols=tuple(range(1, 9)), dtype="i")
    cells -= 1  # vertices numbering start at 0
    return vertices, cells, cell_layer


def check_mesh(vertices, cells, cell_layer):
    xy = vertices[:, :2]
    bottom_left = xy.min(axis=0)
    upper_right = xy.max(axis=0)
    Lx, Ly = upper_right - bottom_left
    x, y, z = (vertices[:, j] for j in range(3))
    base = np.transpose(cells[:, :4])
    top = np.transpose(cells[:, 4:])
    assert all([np.all(x[base[i]] == x[top[i]]) for i in range(4)])
    assert all([np.all(y[base[i]] == y[top[i]]) for i in range(4)])
    assert all([np.all(z[base[i]] <= z[top[i]]) for i in range(4)])
    dx = x[cells[:, 1]] - x[cells[:, 0]]
    dy = y[cells[:, 2]] - y[cells[:, 0]]
    dx = np.unique(dx)
    dy = np.unique(dy)
    dxmin = dx.min()
    dymin = dy.min()
    assert dxmin == dymin
    ix = (x - bottom_left[0]) / dxmin
    assert all(np.modf(ix)[0] == 0)
    ix = np.array(ix, dtype="i")
    jy = (y - bottom_left[1]) / dymin
    assert all(np.modf(jy)[0] == 0)
    jy = np.array(jy, dtype="i")
    print("grid loaded")
    nb_inoeuds = int(1 + Lx / dxmin)
    nb_jnoeuds = int(1 + Ly / dymin)
    coord = [ix, jy, x, y, z]
    return dxmin, dymin, nb_inoeuds, nb_jnoeuds, coord, bottom_left, upper_right


def submesh(vertices, cells, cell_layer, center, radius):
    # on utilise que les 2 premiers colonnes (x et y)
    keep = np.linalg.norm(vertices[:, :2] - center[:2], axis=1) < radius
    kept_vertices = vertices[keep]
    kept_nodes = np.nonzero(keep)[0]
    print("on garde", kept_nodes.shape[0], "noeuds")
    keep = np.ones(cells.shape[0], dtype=np.bool)
    for j in range(cells.shape[1]):
        keep &= np.in1d(cells[:, j], kept_nodes)
    kept_cells = cells[keep]
    kept_cell_layer = cell_layer[keep]
    print("on garde", kept_cells.shape[0], "cellules")
    kept_nodes_id = np.empty(vertices.shape[0], dtype="i")
    kept_nodes_id[kept_nodes] = np.arange(kept_nodes.shape[0])
    kept_cells = (lambda i: kept_nodes_id[i])(kept_cells)
    return kept_vertices, kept_cells, kept_cell_layer


def collect_layer_nodes(cells, cell_layer):
    # trouve les noeuds de chaque layer
    layer_nodes = []
    id_layers = np.unique(cell_layer)
    # la 1ere couche est la plus haute on va donc stocker
    # les noeuds des faces du haut des cellules dans cette couche
    # (c'est à dire les 4 dernière colonnes)
    # donc les deux premieres surfaces ont la meme "forme"/carte
    layer_nodes.append(cells[cell_layer == id_layers[0], 4:])
    for li in id_layers:
        # on recupere les 4 premieres colonnes (face du bas)
        # pour chaque cellule dans le layer li
        layer_nodes.append(cells[cell_layer == li, :4])
    return layer_nodes


def collect_tops(cells, cell_layer):
    layer_nodes = []
    id_layers = np.unique(cell_layer)
    for li in id_layers:
        layer_nodes.append(cells[cell_layer == li, 4:])
    return layer_nodes


def collect_bases(cells, cell_layer):
    layer_nodes = []
    id_layers = np.unique(cell_layer)
    for li in id_layers:
        layer_nodes.append(cells[cell_layer == li, :4])
    return layer_nodes


def compute_surfaces(
    nb_layers, Lx, dxmin, Ly, dymin, bottom_left, cells, cell_layer, x, y, z, position
):
    surfaces = np.empty(
        (nb_layers, int(1 + Lx / dxmin), int(1 + Ly / dymin)), dtype="d"
    )
    surfaces[:] = 9999.0
    ix = (x - bottom_left[0]) / dxmin
    assert all(np.modf(ix)[0] == 0)
    ix = np.array(ix, dtype="i")
    jy = (y - bottom_left[1]) / dymin
    assert all(np.modf(jy)[0] == 0)
    jy = np.array(jy, dtype="i")

    if position == "top":
        layer_nodes = collect_tops(cells, cell_layer)
    if position == "base":
        layer_nodes = collect_bases(cells, cell_layer)

    for surface, squares in zip(surfaces, layer_nodes):
        nodes = np.unique(squares)
        surface[ix[nodes], jy[nodes]] = z[nodes]
        not_small = ix[squares[:, 1]] - ix[squares[:, 0]] > 1
        for square in squares[not_small]:
            f = interpolate.interp2d(ix[square], jy[square], z[square], kind="linear")
            imin, imax = ix[square[0]], ix[square[1]] + 1
            jmin, jmax = jy[square[0]], jy[square[2]] + 1
            zinterp = f(np.arange(imin, imax), np.arange(jmin, jmax))
            surface[imin:imax, jmin:jmax] = np.transpose(zinterp)
    return surfaces


def compute_thickness(vertices, cells, x, y, z):
    base = np.transpose(cells[:, :4])
    top = np.transpose(cells[:, 4:])
    thickness = np.vstack([z[top[i]] - z[base[i]] for i in range(4)])
    return thickness.mean(axis=0)


def compute_MNT(vertices, cells, cell_layer, nb_inoeuds, nb_jnoeuds, coord):
    ix, jy, x, y, z = coord
    id_layers = np.unique(cell_layer)
    nb_layers = id_layers.shape[0]
    surfaces = np.empty((nb_layers + 1, nb_inoeuds, nb_jnoeuds), dtype="d")
    surfaces[:] = -9999.0
    # size_masks = [X[cells[:,1]] - X[cells[:,0]] == dxi for dxi in dx]
    # small_cell, medium_cell, large_cell = size_masks
    layer_nodes = collect_layer_nodes(cells, cell_layer)
    for surface, squares in zip(surfaces, layer_nodes):
        nodes = np.unique(squares)
        surface[ix[nodes], jy[nodes]] = z[nodes]
        not_small = ix[squares[:, 1]] - ix[squares[:, 0]] > 1
        for square in squares[not_small]:
            f = interpolate.interp2d(ix[square], jy[square], z[square], kind="linear")
            imin, imax = ix[square[0]], ix[square[1]] + 1
            jmin, jmax = jy[square[0]], jy[square[2]] + 1
            zinterp = f(np.arange(imin, imax), np.arange(jmin, jmax))
            surface[imin:imax, jmin:jmax] = np.transpose(zinterp)
    MNT = surfaces.max(axis=0)
    #    masked_MNT = np.ma.masked_array(MNT, MNT>9998)
    return MNT


def compute_thickness_layers(
    vertices, cells, cell_layer, nb_inoeuds, nb_jnoeuds, coord, bottom_left
):
    ix, jy, x, y, z = coord
    #    thick = [[z[top[i,j]]-z[base[i,j]] for i in range(base.shape[0])] for j in range(base.shape[1])]
    #    thick = [y for x in thick for y in x] #flattened
    # Creation de 20 couches contenant les épaisseurs de chaque layer
    #    thick =np.array(thick).flatten()
    id_layers = np.unique(cell_layer)
    nb_layers = id_layers.shape[0]
    dx = np.unique(x[cells[:, 1]] - x[cells[:, 0]])
    layers_thickness = np.empty((nb_layers, nb_inoeuds - 1, nb_jnoeuds - 1), dtype="d")
    layers_thickness[:] = 0
    nx, ny = layers_thickness.shape[1:]
    for k, lk in enumerate(id_layers):
        layer_cells = cells[cell_layer == lk]
        for c in range(layer_cells.shape[0]):
            node1 = layer_cells[c][0]
            node2 = layer_cells[c][1]
            size = vertices[node2][0] - vertices[node1][0]
            i = (vertices[node1][0] - bottom_left[0]) / dx[0]
            j = (vertices[node1][1] - bottom_left[1]) / dx[0]
            layers_thickness[k, int(i), int(j)] = 1
            if size == dx[1]:
                layers_thickness[k, int(i + 1), int(j)] = 1
                layers_thickness[k, int(i), int(j + 1)] = 1
                layers_thickness[k, int(i + 1), int(j + 1)] = 1
            if size == dx[2]:
                for a in range(4):
                    for b in range(4):
                        layers_thickness[k, int(i + a), int(j + b)] = 1
    return layers_thickness


def dummy_mesh():
    vertices = np.array(
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 3, 0),
            (1, 3, 0),
            (0, 4, 0),
            (1, 4, 0),
            (3, 0, 0),
            (1, 2, 0),
            (3, 2, 0),
            (7, 0, 0),
            (3, 4, 0),
            (7, 4, 0),
            (0, 0, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 2),
            (0, 3, 1),
            (1, 3, 1),
            (0, 4, 1),
            (1, 4, 1),
            (3, 0, 0),
            (1, 2, 2),
            (3, 2, 2),
            (7, 0, 6),
            (3, 4, 6),
            (7, 4, 12),
            (0, 0, 1),
            (1, 0, 2),
            (0, 1, 2),
            (1, 1, 3),
            (0, 3, 2),
            (1, 3, 2),
            (0, 4, 2),
            (1, 4, 2),
            (3, 0, 1),
            (1, 2, 3),
            (3, 2, 3),
            (7, 0, 7),
            (3, 4, 7),
            (7, 4, 13),
        ],
        dtype="d",
    )
    cells = np.array(
        [
            (0, 1, 2, 3, 14, 15, 16, 17),
            (4, 5, 6, 7, 18, 19, 20, 21),
            (1, 8, 9, 10, 15, 22, 23, 24),
            (8, 11, 12, 13, 22, 25, 26, 27),
            (14, 15, 16, 17, 28, 29, 30, 31),
            (18, 19, 20, 21, 32, 33, 34, 35),
            (15, 22, 23, 24, 29, 36, 37, 38),
            (22, 25, 26, 27, 36, 39, 40, 41),
        ],
        dtype="i",
    )
    cell_layer = np.zeros(cells.shape[0], dtype="i")
    for i in range(int(cells.shape[0] / 2)):
        cell_layer[int(cells.shape[0] / 2) + i] = 1
    # Pour que les layers soient classés vers le bas
    cell_layer = cell_layer.max() - cell_layer
    return vertices, cells, cell_layer


def dummy_mesh2():
    vertices = np.array(
        [
            (0, 0, 0),
            (2, 0, 0),
            (0, 2, 0),
            (2, 2, 0),
            (3, 0, 0),
            (2, 1, 0),
            (3, 1, 0),
            (5, 0, 0),
            (3, 2, 0),
            (5, 2, 0),
            (0, 0, 1),
            (2, 0, 1),
            (0, 2, 2),
            (2, 2, 2),
            (3, 0, 1),
            (2, 1, 0.5),
            (3, 1, 2),
            (5, 0, 2),
            (3, 2, 1),
            (5, 2, 0.5),
        ],
        dtype="d",
    )
    cells = np.array(
        [
            (0, 1, 2, 3, 10, 11, 12, 13),
            (1, 4, 5, 6, 11, 14, 15, 16),
            (4, 7, 8, 9, 14, 17, 18, 19),
        ],
        dtype="i",
    )
    cell_layer = np.zeros(cells.shape[0], dtype="i")
    return vertices, cells, cell_layer


class ArretesNulles(object):
    def __init__(self, cols, vertices, cells, layers, affls, bots):
        x, y, z = (vertices[:, j] for j in range(3))
        mask_cells = np.zeros((cells.shape[0], 4))
        for i in range(4):
            mask_cells[z[cells[:, i]] == z[cells[:, i + 4]], i] = 1
        mask_cells = np.sum(mask_cells, axis=1)
        # indices = np.where(mask_cells == 3)
        # mask_cells[indices[0][5]] = 999.
        # mask_cells = np.where(mask_cells != 999., 0, mask_cells)
        # mask_cells = np.where(mask_cells == 999., 3., mask_cells)
        self.x, self.y, self.z = x, y, z
        self.affls = affls
        self.bots = bots
        self.layers = layers
        self.cells = cells
        self.vertices = vertices
        self.cols = cols
        self.mask_cells = mask_cells
        self.cells_nodes = {"wedges": [], "pyramids": [], "tetras": [], "hexas": []}
        self.cells_layers = {"wedges": [], "pyramids": [], "tetras": [], "hexas": []}
        self.cells_affls = {"wedges": [], "pyramids": [], "tetras": [], "hexas": []}
        self.cells_bots = {"wedges": [], "pyramids": [], "tetras": [], "hexas": []}

    def _get_faces_bot(self, faces_bots, faces_four, faces_three):
        z_vertices = self.vertices[faces_four][:, :, :, -1]
        z_mean = np.mean(z_vertices, axis=-1)
        z_vertices = self.vertices[faces_three][:, :, :, -1]
        z_mean_2 = np.mean(z_vertices, axis=-1)
        z_mean = np.concatenate([z_mean, z_mean_2], axis=1)
        indices = np.argmin(z_mean, axis=1)
        toto = np.indices(z_mean.shape)
        uu, vv = np.where(toto[1] == np.repeat(indices[:, np.newaxis], 5, axis=-1))
        faces_bots = faces_bots[uu, vv]
        return faces_bots

    def _get_faces_affl(self, faces_bots, faces_four, faces_three):
        z_vertices = self.vertices[faces_four][:, :, :, -1]
        z_mean = np.mean(z_vertices, axis=-1)
        z_vertices = self.vertices[faces_three][:, :, :, -1]
        z_mean_2 = np.mean(z_vertices, axis=-1)
        z_mean = np.concatenate([z_mean, z_mean_2], axis=1)
        indices = np.argmax(z_mean, axis=1)
        toto = np.indices(z_mean.shape)
        uu, vv = np.where(toto[1] == np.repeat(indices[:, np.newaxis], 5, axis=-1))
        faces_bots = faces_bots[uu, vv]
        return faces_bots

    def set_cells(self):
        for mesh in ["wedges", "pyramids", "tetras"]:
            self.cells_nodes[mesh] = np.concatenate(
                self.cells_nodes[mesh], axis=0
            ).tolist()
            self.cells_layers[mesh] = np.concatenate(
                self.cells_layers[mesh], axis=0
            ).tolist()
            self.cells_affls[mesh] = np.concatenate(
                self.cells_affls[mesh], axis=0
            ).tolist()
            self.cells_bots[mesh] = np.concatenate(
                self.cells_bots[mesh], axis=0
            ).tolist()
        self.cells_nodes["hexas"].extend(self.cells[self.mask_cells == 0].tolist())
        self.cells_layers["hexas"].extend(self.layers[self.mask_cells == 0].tolist())
        self.cells_affls["hexas"].extend(self.affls[self.mask_cells == 0].tolist())
        self.cells_bots["hexas"].extend(self.bots[self.mask_cells == 0].tolist())
        self.compute_faces()

    def compute_faces(self):
        #  Faces 4 noeuds
        newcells_faces = {}
        newfaces_nodes = []
        indices = {
            "hexas": [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [3, 2, 6, 7],
                [0, 3, 7, 4],
                [1, 2, 6, 5],
            ],
            "wedges": [[0, 2, 5, 3], [1, 4, 5, 2], [0, 1, 4, 3]],
            "pyramids": [[0, 1, 2, 3]],
        }
        faces, nmin, nmax = [], [], []
        for mesh in ["hexas", "wedges", "pyramids"]:
            nmin.append(len(faces))
            for i, indice in enumerate(indices[mesh]):
                faces.extend(np.array(self.cells_nodes[mesh])[:, indice].tolist())
            nmax.append(len(faces))
        faces_sort = np.sort(faces, axis=1)
        newfaces, unique_indices, face_id = np.unique(
            faces_sort, axis=0, return_inverse=True, return_index=True
        )
        max_face_id = np.max(face_id) + 1
        newfaces_nodes.extend(np.array(faces)[unique_indices].tolist())
        for i, mesh in enumerate(["hexas", "wedges", "pyramids"]):
            newcells_faces[mesh] = face_id[nmin[i] : nmax[i]].reshape(
                (len(self.cells_nodes[mesh]), len(indices[mesh])), order="F"
            )
        #  Faces 3 noeuds
        indices = {
            "wedges": [[0, 1, 2], [3, 4, 5]],
            "tetras": [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
            "pyramids": [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        }
        faces, nmin, nmax = [], [], []
        for mesh in ["wedges", "pyramids", "tetras"]:
            nmin.append(len(faces))
            for i, indice in enumerate(indices[mesh]):
                faces.extend(np.array(self.cells_nodes[mesh])[:, indice].tolist())
            nmax.append(len(faces))
        faces_sort = np.sort(faces, axis=1)
        newfaces, unique_indices, face_id = np.unique(
            faces_sort, axis=0, return_inverse=True, return_index=True
        )
        face_id = max_face_id + face_id
        newfaces_nodes.extend(np.array(faces)[unique_indices].tolist())
        for i, mesh in enumerate(["wedges", "pyramids", "tetras"]):
            if mesh in newcells_faces.keys():
                toto = face_id[nmin[i] : nmax[i]].reshape(
                    (len(self.cells_nodes[mesh]), len(indices[mesh])), order="F"
                )
                newcells_faces[mesh] = np.concatenate(
                    [newcells_faces[mesh], toto], axis=1
                )
            else:
                newcells_faces[mesh] = face_id[nmin[i] : nmax[i]].reshape(
                    (len(self.cells_nodes[mesh]), len(indices[mesh])), order="F"
                )
        self.faces_nodes = newfaces_nodes
        self.cells_faces = newcells_faces
        self.set_faces_nodes_bot(max_face_id)
        self.set_faces_nodes_affl(max_face_id)

    def set_faces_nodes_bot(self, max_face_id):
        self.faces_nodes_bot = {}
        newfaces_nodes = np.array(self.faces_nodes)
        four_nodes = np.array(self.faces_nodes[:max_face_id])
        # --- hexas
        cells_bots = np.array(self.cells_bots["hexas"])
        faces_bots = self.cells_faces["hexas"][cells_bots != 0][:, 0]
        self.faces_nodes_bot["hexas"] = four_nodes[faces_bots].tolist()
        # --- wedges
        cells_bots = np.array(self.cells_bots["wedges"])
        faces_bots = self.cells_faces["wedges"][cells_bots != 0]
        faces_four = four_nodes[faces_bots[:, :3]]
        faces_three = np.array(newfaces_nodes[faces_bots[:, 3:]].tolist())
        faces_bots = self._get_faces_bot(faces_bots, faces_four, faces_three)
        self.faces_nodes_bot["wedges"] = newfaces_nodes[faces_bots].tolist()
        # --- pyramids
        cells_bots = np.array(self.cells_bots["pyramids"])
        faces_bots = self.cells_faces["pyramids"][cells_bots != 0]
        faces_four = four_nodes[faces_bots[:, :1]]
        faces_three = np.array(newfaces_nodes[faces_bots[:, 1:]].tolist())
        faces_bots = self._get_faces_bot(faces_bots, faces_four, faces_three)
        self.faces_nodes_bot["pyramids"] = newfaces_nodes[faces_bots].tolist()
        # --- tetras
        cells_bots = np.array(self.cells_bots["tetras"])
        faces_bots = self.cells_faces["tetras"][cells_bots != 0]
        faces_nodes = np.array(newfaces_nodes[faces_bots].tolist())
        z_vertices = self.vertices[faces_nodes][:, :, :, -1]
        z_mean = np.mean(z_vertices, axis=-1)
        indices = np.argmin(z_mean, axis=1)
        toto = np.indices(z_mean.shape)
        uu, vv = np.where(toto[1] == np.repeat(indices[:, np.newaxis], 4, axis=-1))
        faces_bots = faces_bots[uu, vv]
        self.faces_nodes_bot["tetras"] = newfaces_nodes[faces_bots].tolist()

    def set_faces_nodes_affl(self, max_face_id):
        self.faces_nodes_affl = {}
        newfaces_nodes = np.array(self.faces_nodes)
        four_nodes = np.array(self.faces_nodes[:max_face_id])
        # --- hexas
        cells_affls = np.array(self.cells_affls["hexas"])
        faces_affls = self.cells_faces["hexas"][cells_affls != 0][:, 1]
        self.faces_nodes_affl["hexas"] = four_nodes[faces_affls].tolist()
        # --- wedges
        cells_affls = np.array(self.cells_affls["wedges"])
        faces_affls = self.cells_faces["wedges"][cells_affls != 0]
        faces_four = four_nodes[faces_affls[:, :3]]
        faces_three = np.array(newfaces_nodes[faces_affls[:, 3:]].tolist())
        faces_affls = self._get_faces_affl(faces_affls, faces_four, faces_three)
        self.faces_nodes_affl["wedges"] = newfaces_nodes[faces_affls].tolist()
        # --- pyramids
        cells_affls = np.array(self.cells_affls["pyramids"])
        faces_affls = self.cells_faces["pyramids"][cells_affls != 0]
        faces_four = four_nodes[faces_affls[:, :1]]
        faces_three = np.array(newfaces_nodes[faces_affls[:, 1:]].tolist())
        faces_affls = self._get_faces_affl(faces_affls, faces_four, faces_three)
        self.faces_nodes_affl["pyramids"] = newfaces_nodes[faces_affls].tolist()
        # --- tetras
        cells_affls = np.array(self.cells_affls["tetras"])
        faces_affls = self.cells_faces["tetras"][cells_affls != 0]
        faces_nodes = np.array(newfaces_nodes[faces_affls].tolist())
        z_vertices = self.vertices[faces_nodes][:, :, :, -1]
        z_mean = np.mean(z_vertices, axis=-1)
        indices = np.argmax(z_mean, axis=1)
        toto = np.indices(z_mean.shape)
        uu, vv = np.where(toto[1] == np.repeat(indices[:, np.newaxis], 4, axis=-1))
        faces_affls = faces_affls[uu, vv]
        self.faces_nodes_affl["tetras"] = newfaces_nodes[faces_affls].tolist()

    def _add_vertices(self, masks):
        nvert = self.vertices.shape[0]
        tvertices, mailles = [], []
        for m in masks:
            if np.any(m):
                mailles_temp = []
                xmean = np.mean(self.x[self.cells[m]], axis=1)
                ymean = np.mean(self.y[self.cells[m]], axis=1)
                zmean_bas = np.round(np.mean(self.z[self.cells[m]][:, :4], axis=1), 2)
                zmean_haut = np.round(np.mean(self.z[self.cells[m]][:, 4:], axis=1), 2)
                xyz_bas = np.vstack([xmean, ymean, zmean_bas]).T.tolist()
                xyz_haut = np.vstack([xmean, ymean, zmean_haut]).T.tolist()
                for i in range(self.cells[m].shape[0]):
                    if xyz_bas[i] not in tvertices:
                        tvertices.append(xyz_bas[i])
                        imaille_bas = nvert + len(tvertices) - 1
                    else:
                        imaille_bas = nvert + tvertices.index(xyz_bas[i])
                    if xyz_haut[i] not in tvertices:
                        tvertices.append(xyz_haut[i])
                        imaille_haut = nvert + len(tvertices) - 1
                    else:
                        imaille_haut = nvert + tvertices.index(xyz_haut[i])
                    mailles_temp.append([imaille_bas, imaille_haut])
                mailles.append(np.array(mailles_temp))
        return tvertices, mailles

    def _add_wedges(self, mailles, mask):
        if np.any(mask):
            iw = [(0, 1, 4, 5), (3, 2, 7, 6), (0, 3, 4, 7), (1, 2, 5, 6)]
            for j in iw:
                self.cells_nodes["wedges"].append(
                    np.hstack(
                        [
                            self.cells[mask][:, [j[0], j[1]]],
                            mailles[0][:, [0]],
                            self.cells[mask][:, [j[2], j[3]]],
                            mailles[0][:, [1]],
                        ]
                    )
                )
                self.cells_layers["wedges"].append(self.layers[mask])
                self.cells_affls["wedges"].append(self.affls[mask])
                self.cells_bots["wedges"].append(self.bots[mask])

    def _get_mask(self, num):
        ones = self.mask_cells == num
        mask = [col in self.cols[ones] for col in self.cols]
        for inum in range(1, 5):
            mask = np.where(self.mask_cells == inum, False, mask)
        mask = np.where(self.mask_cells == -1, False, mask)
        assert np.all(self.mask_cells[mask] == 0)
        self.mask_cells[ones] = -1
        self.mask_cells[mask] = -1
        cellsn = self.cells[ones]
        return mask, ones, cellsn

    def one_vertice(self):
        zeros, ones, cells_ones = self._get_mask(1)
        tvertices, mailles = self._add_vertices([zeros, ones])
        self._add_wedges(mailles, zeros)
        dic_iw = [
            [(3, 2, 7, 6), (1, 2, 5, 6)],
            [(0, 3, 4, 7), (3, 2, 7, 6)],
            [(0, 1, 4, 5), (0, 3, 4, 7)],
            [(0, 1, 4, 5), (1, 2, 5, 6)],
        ]
        dic_ip = [
            [(1, 5, 0), (3, 7, 0)],
            [(0, 4, 1), (2, 6, 1)],
            [(1, 5, 2), (3, 7, 2)],
            [(0, 4, 3), (2, 6, 3)],
        ]
        for i in range(4):
            m = self.z[cells_ones[:, i]] == self.z[cells_ones[:, i + 4]]
            if np.any(m):
                for j in dic_iw[i]:
                    self.cells_nodes["wedges"].append(
                        np.hstack(
                            [
                                cells_ones[m][:, [j[0], j[1]]],
                                mailles[1][m][:, [0]],
                                cells_ones[m][:, [j[2], j[3]]],
                                mailles[1][m][:, [1]],
                            ]
                        )
                    )
                    self.cells_layers["wedges"].append(self.layers[ones][m])
                    self.cells_affls["wedges"].append(self.affls[ones][m])
                    self.cells_bots["wedges"].append(self.bots[ones][m])
                for j in dic_ip[i]:
                    self.cells_nodes["pyramids"].append(
                        np.hstack(
                            [
                                cells_ones[m][:, [j[0]]],
                                mailles[1][m][:, [0, 1]],
                                cells_ones[m][:, [j[1], j[2]]],
                            ]
                        )
                    )
                    self.cells_layers["pyramids"].append(self.layers[ones][m])
                    self.cells_affls["pyramids"].append(self.affls[ones][m])
                    self.cells_bots["pyramids"].append(self.bots[ones][m])
        if tvertices:
            self.vertices = np.concatenate([self.vertices, tvertices])

    def two_vertices(self):
        mask = self.mask_cells == 2
        if np.any(mask):
            self.cells_layers["wedges"].append(self.layers[mask])
            self.cells_affls["wedges"].append(self.affls[mask])
            self.cells_bots["wedges"].append(self.bots[mask])
            tcoins = self.cells[mask]
            self.mask_cells[mask] = -1
            wedges = []
            cells_wedges = {
                (0, 1): [3, 4, 7, 2, 5, 6],
                (0, 3): [1, 4, 5, 2, 7, 6],
                (1, 2): [3, 6, 7, 0, 5, 4],
                (3, 2): [0, 3, 4, 1, 2, 5],
            }
            positions = [(0, 2), (3, 1)]
            for pos in positions:
                assert not any(
                    np.all(
                        self.z[tcoins[:, pos]]
                        == self.z[tcoins[:, [pos[0] + 4, pos[1] + 4]]],
                        axis=1,
                    )
                )
            positions = [(0, 1), (0, 3), (3, 2), (1, 2)]
            for pos in positions:
                m = np.all(
                    self.z[tcoins[:, pos]]
                    == self.z[tcoins[:, [pos[0] + 4, pos[1] + 4]]],
                    axis=1,
                )
                iw = cells_wedges[pos]
                wedges.append(tcoins[m][:, iw])
            if wedges:
                self.cells_nodes["wedges"].extend(wedges)

    def three_vertices(self):
        zeros, threes, cells_threes = self._get_mask(3)
        tvertices, mailles = self._add_vertices([zeros, threes])
        self._add_wedges(mailles, zeros)
        dic_it = [
            [(3, 2), (1, 2)],
            [(0, 3), (3, 2)],
            [(0, 3), (0, 1)],
            [(0, 1), (1, 2)],
        ]
        dic_ip = [
            [(0, 4, 1), (0, 4, 3)],
            [(1, 5, 0), (1, 5, 2)],
            [(2, 6, 3), (2, 6, 1)],
            [(3, 7, 0), (3, 7, 2)],
        ]
        for i in range(4):
            m = self.z[cells_threes[:, i]] != self.z[cells_threes[:, i + 4]]
            if np.any(m):
                for j in dic_it[i]:
                    self.cells_nodes["tetras"].append(
                        np.hstack(
                            [mailles[1][m][:, [0, 1]], cells_threes[m][:, [j[0], j[1]]]]
                        )
                    )
                    self.cells_layers["tetras"].append(self.layers[threes][m])
                    self.cells_affls["tetras"].append(self.affls[threes][m])
                    self.cells_bots["tetras"].append(self.bots[threes][m])
                for j in dic_ip[i]:
                    self.cells_nodes["pyramids"].append(
                        np.hstack(
                            [
                                cells_threes[m][:, [j[0]]],
                                mailles[1][m][:, [0, 1]],
                                cells_threes[m][:, [j[1], j[2]]],
                            ]
                        )
                    )
                    self.cells_layers["pyramids"].append(self.layers[threes][m])
                    self.cells_affls["pyramids"].append(self.affls[threes][m])
                    self.cells_bots["pyramids"].append(self.bots[threes][m])
        if tvertices:
            self.vertices = np.concatenate([self.vertices, tvertices])
