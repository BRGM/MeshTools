from MeshTools.edgeutils import *

e1 = Edge(6, 4)
print("Edge are ordered:", e1)
e2 = Edge(7, 4)
print(e1, "<", e2, ":", e1 < e2)

d = {}
d[e1] = "e1"
d[e2] = "e2"
print(d)

faces = [[0, 1, 2, 3], [2, 3, 4, 5]]
fmap = Edge_faces()
fmap.register_faces(faces)

for e in [Edge(0, 1), Edge(2, 3)]:
    print(e, "is shared by faces:", fmap.edge_faces[e])
    assert all(edge_is_in_face(e, faces[fi]) for fi in fmap.edge_faces[e])

face = [0, 2, 3, 5, 4]
print("original face:", face)
new_edges = [Edge(5, 7), Edge(6, 7), Edge(4, 6)]
face = replace_edge(face, Edge(5, 4), new_edges)
print("face with replaced edge:", face)

face = [0, 2, 3, 5, 4]
print("original face:", face)
new_edges = [Edge(12, 0), Edge(12, 13), Edge(13, 14), Edge(14, 2)]
face = replace_edge(face, Edge(2, 0), new_edges)
print("face with replaced edge:", face)
