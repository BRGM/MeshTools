from itertools import chain, product
import numpy as np

import MeshTools as MT
from MeshTools import idarray, HexMesh
from MeshTools.GridTools import grid2hexs
import MeshTools.RawMesh as RM
import MeshTools.edgeutils as edge

# intial grid shape
grid_shape = (2, 1, 1)
# number of edges to be splitted randomly
nb_split = 20
np.random.seed(12345)

vertices, hexahedron = grid2hexs(shape=grid_shape, extent=grid_shape)
mesh = HexMesh.make(vertices, idarray(hexahedron))
_, face_nodes = mesh.faces_nodes_as_COC()
face_nodes = face_nodes.array_view()
face_nodes.shape = (-1, 4)
face_nodes = list(face_nodes)
_, cell_faces = mesh.cells_faces_as_COC()
cell_faces = cell_faces.array_view()
cell_faces.shape = (-1, 6)
cell_faces = list(cell_faces)

hexahedron_centers = np.array([np.mean(vertices[cell], axis=0) for cell in hexahedron])
quadrangle_centers = np.array([np.mean(vertices[face], axis=0) for face in face_nodes])

edge_map = edge.Edge_faces(face_nodes)
assert edge_map.seems_consistent()

vertices = list(vertices)
edges = list(edge_map.edges())
edge_parts = [[] for k in range(edge_map.nb_edges)]


def draw_splittable_edge():
    k = int(len(edge_parts) * np.random.rand())
    candidates = edge_parts[k]
    while len(candidates) > 0:
        assert len(candidates) > 1
        k = candidates[int(len(candidates) * np.random.rand())]
        candidates = edge_parts[k]
    return k


def share_node(e1, e2):
    return any(n in e2 for n in e1)


def shared_node(e1, e2):
    for i, j in product((0, 1), (0, 1)):
        if e1[i] == e2[j]:
            return e1[i]


def is_chain(edges):
    return all(share_node(edges[i - 1], edges[i]) for i in range(1, len(edges)))


def hook_head(head, parts):
    if head not in edges[parts[0]]:
        parts.reverse()
    assert head in edges[parts[0]]
    return parts


def elementary_partition(e, parts):
    assert len(parts) > 1
    # print('Decomposing:', e, '->', parts)
    head, tail = e
    parts = hook_head(head, parts)
    assert tail in edges[parts[-1]]
    while not all(len(edge_parts[k]) == 0 for k in parts):
        subparts = [[k] if len(edge_parts[k]) == 0 else edge_parts[k] for k in parts]
        assert all(is_chain([edges[k] for k in part]) for part in subparts)
        if len(subparts[0]) > 1:
            subparts[0] = hook_head(head, subparts[0])
        assert head in edges[subparts[0][0]]
        for i in range(1, len(subparts)):
            if not share_node(edges[subparts[i - 1][-1]], edges[subparts[i][0]]):
                subparts[i].reverse()
            assert share_node(edges[subparts[i - 1][-1]], edges[subparts[i][0]])
        assert tail in edges[subparts[-1][-1]]
        parts = list(chain.from_iterable(subparts))
        # print('    substep:', parts)
    # print('Final partition:', parts)
    # print('                ', [edges[k] for k in parts])
    return [edges[k] for k in parts]


def split_edge(k):
    parts = edge_parts[k]
    assert len(parts) == 0
    n1, n2 = edges[k]
    A, B = vertices[n1], vertices[n2]
    # print('split edge', k, 'with nodes', (n1, n2), 'and vertices', A, B)
    M = np.mean([A, B], axis=0)
    nM = len(vertices)
    vertices.append(M)
    k = len(edges)
    edges.append(edge.Edge(n1, nM))
    edges.append(edge.Edge(n2, nM))
    parts.extend((k, k + 1))
    edge_parts.extend([[], []])


for _ in range(nb_split):
    split_edge(draw_splittable_edge())

# this is mandatory for further computations
vertices = np.array(vertices)

assert all(is_chain([edges[k] for k in part]) for part in edge_parts)

splitted_edges = [
    (k, edges[k], elementary_partition(edges[k], edge_parts[k]))
    for k, parts in enumerate(edge_parts[: edge_map.nb_edges])
    if len(parts) != 0
]

for _, old_edge, new_edges in splitted_edges:
    for face in edge_map.edge_faces[old_edge]:
        face_nodes[face] = edge.replace_edge(face_nodes[face], old_edge, new_edges)

face_nodes = [np.array(nodes, copy=False, dtype=np.int) for nodes in face_nodes]

mesh = RM.RawMesh(vertices=vertices, face_nodes=face_nodes, cell_faces=cell_faces)
raw_mesh, original_cell = mesh.as_hybrid_mesh(
    convert_voxels=True,
    cell_centers=hexahedron_centers,
    face_centers=quadrangle_centers,
)
MT.to_vtu(
    raw_mesh,
    "splitted_grid_edges",
    celldata={"original_cell": original_cell},
)
