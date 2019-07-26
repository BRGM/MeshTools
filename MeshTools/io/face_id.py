import numpy as np

def sort_face_nodes(nodes):
    nodes = np.asarray(nodes)
    assert nodes.ndim==1
    assert nodes.shape[0]>2
    nimin = np.argmin(nodes)
    nodes = np.hstack([nodes[nimin:], nodes[:nimin]])
    if nodes[1]<nodes[-1]:
        return nodes
    return np.hstack([nodes[0], nodes[1:][::-1]])

def is_same_face(nodes1, nodes2):
    return np.all(sort_face_nodes(nodes1)==sort_face_nodes(nodes2))

if __name__=='__main__':
    nodes1 = [3, 6, 5, 2, 4]
    nodes2 = [2, 5, 6, 3, 4]
    print(sort_face_nodes(nodes1))
    print(sort_face_nodes(nodes2))
    print(is_same_face(nodes1, nodes2))



