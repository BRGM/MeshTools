import os
import numpy as np
import MeshTools as MT

class RawMesh:

    def __init__(self, **kwds):
        assert 'vertices' in kwds
        #assert 'cell_nodes' in kwds
        self.cell_nodes = None
        assert 'cell_faces' in kwds
        assert 'face_nodes' in kwds
        for name, value in kwds.items():
            setattr(self, name, value)
    
    @classmethod
    def convert(cls, other):
        info = { name: getattr(other, name)
                 for name in ['vertices', 'cell_faces', 'face_nodes'] }
        if hasattr(other, 'cell_nodes'):
            info['cell_nodes'] = getattr(other, 'cell_nodes')
        return cls(**info)

    def collect_cell_nodes(self):
        assert self.cell_nodes is None
        # WARNING this is ok for convex cells
        face_nodes = self.face_nodes
        cell_faces = self.cell_faces
        return [
            np.unique(
                np.hstack([face_nodes[fi] for fi in faces])
            ) for faces in cell_faces
        ]

    def as_tets(self):
        vertices = np.array(self.vertices, dtype=np.double)
        assert len(vertices.shape)==2
        assert vertices.shape[1]==3
        face_nodes = self.face_nodes
        cell_faces = self.cell_faces
        cell_nodes =  ( self.cell_nodes if self.cell_nodes
                        else self.collect_cell_nodes() )
        cell_centers = np.array([
            np.mean([vertices[n] for n in nodes], axis=0)
            for nodes in cell_nodes
        ])
        nb_nodes = vertices.shape[0]
        nb_cells = cell_centers.shape[0]
        assert nb_nodes>0 # so we can use 0 index as a flag
        face_centers = np.zeros(len(face_nodes), dtype=np.int64)
        additional_vertices = []
        for fi, nodes in enumerate(face_nodes):
            assert len(nodes)>2
            # split faces that are not triangles
            if len(nodes)>3:
                face_centers[fi] = len(additional_vertices)
                additional_vertices.append(
                    np.mean([vertices[n] for n in nodes], axis=0)
                )
        face_centers[face_centers>0]+= nb_nodes + nb_cells
        vertices = np.vstack([vertices, cell_centers, additional_vertices])
        # split cells into tetrahedra (even if they are tetrahedra)
        # WARNING: no check is made on cell geometry
        splitted_cells = []
        for ci, faces in enumerate(cell_faces):
            tets = []
            cc = nb_nodes + ci # cell center index
            assert len(faces)>3
            for face in faces:
                nodes = face_nodes[face]
                if len(nodes)==3:
                    assert face_centers[face]==0
                    tets.append(tuple(nodes) + (cc,))
                else:
                    fc = face_centers[face]
                    for k in range(len(nodes)):
                        tets.append(
                            (nodes[k-1], nodes[k], fc, cc)
                        )
            splitted_cells.append(np.array(tets))
        # reconstruct the original cell index
        nb_cell_tets = np.array([tets.shape[0] for tets in splitted_cells])
        tmp = np.zeros(np.sum(nb_cell_tets), dtype=MT.idtype())
        tmp[np.cumsum(nb_cell_tets[:-1])] = 1
        original_cell_index = np.cumsum(tmp)
        return MT.TetMesh.make(
            vertices, MT.idarray(np.vstack(splitted_cells)),
        ), original_cell_index   
