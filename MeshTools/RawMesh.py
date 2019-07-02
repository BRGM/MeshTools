import os
from itertools import chain
import numpy as np
import MeshTools as MT

class RawMesh:

    def __init__(self, **kwds):
        assert 'vertices' in kwds
        self._cell_nodes = None
        assert 'cell_faces' in kwds
        assert 'face_nodes' in kwds
        for name, value in kwds.items():
            if name=='cell_nodes':
                name = '_cell_nodes'
            if name=='vertices':
                value = np.array(value, copy=False)
                assert value.ndim==2 and value.shape[1]==3
            setattr(self, name, value)

    @classmethod
    def convert(cls, other):
        info = { name: getattr(other, name)
                 for name in ['vertices', 'cell_faces', 'face_nodes'] }
        if hasattr(other, 'cell_nodes'):
            info['cell_nodes'] = getattr(other, 'cell_nodes')
        return cls(**info)

    @property
    def nb_vertices(self):
        return self.vertices.shape[0]

    @property
    def nb_faces(self):
        return len(self.face_nodes)

    @property
    def nb_cells(self):
        return len(self.cell_faces)

    def collect_cell_nodes(self):
        assert self._cell_nodes is None
        # WARNING this is ok for convex cells
        face_nodes, cell_faces = self.face_nodes, self.cell_faces
        return [
            np.unique(
                np.hstack([face_nodes[fi] for fi in faces])
            ) for faces in cell_faces
        ]

    @property
    def cell_nodes(self):
        if self._cell_nodes is not None:
            return self._cell_nodes
        else:
            return self.collect_cell_nodes()

    def _specific_faces(self, nbnodes):
        face_nodes = self.face_nodes
        return np.array(
            [len(nodes)==nbnodes for nodes in face_nodes], dtype=np.bool
        )

    def triangle_faces(self):
        return self._specific_faces(3)

    def quadrangle_faces(self):
        return self._specific_faces(4)

    def tetrahedron_cells(self):
        cell_faces = self.cell_faces
        triangles = self.triangle_faces() 
        return np.array([
                # WARNING: no check on geometry consistency is performed
                len(faces)==4 and all(triangles[face] for face in faces)
                for faces in cell_faces
            ], dtype=np.bool
        )

    def hexahedron_cells(self):
        cell_faces = self.cell_faces
        quadrangles = self.quadrangle_faces() 
        return np.array([
                # WARNING: no check on geometry consistency is performed
                len(faces)==6 and all(quadrangles[face] for face in faces)
                for faces in cell_faces
            ], dtype=np.bool
        )

    def _centers(self, element_nodes):
        # CHECKME: returns gravity center, is this the most useful?
        # WARNING: ok for convex elements, no geometry check is performed
        vertices = self.vertices
        assert vertices.ndim==2 and vertices.shape[1]==3
        return np.array([
            np.mean([vertices[n] for n in nodes], axis=0)
            for nodes in element_nodes
        ])

    def cell_centers(self, cell_nodes=None):
        if cell_nodes is None:
            cell_nodes = self.cell_nodes
        return self._centers(cell_nodes)

    def face_centers(self, face_nodes=None):
        if face_nodes is None:
            face_nodes = self.face_nodes
        return self._centers(face_nodes)
    
    def _new_vertices(
        self, cell_centers, kept_cells, face_nodes, kept_faces,
        face_centers=None,
    ):
        splitted_cells = np.logical_not(kept_cells)
        splitted_faces = np.logical_not(kept_faces)
        if face_centers is None:
            face_centers = self.face_centers(
                [face_nodes[fi] for fi in np.nonzero(splitted_faces)[0]]
            )
        else:
            face_centers = face_centers[splitted_faces]
        new_vertices = np.vstack([
            self.vertices,
            np.reshape(cell_centers[splitted_cells], (-1, 3)),
            np.reshape(face_centers, (-1, 3)),
        ])
        # cell center (valid only for splitted cells)
        cc = (self.nb_vertices - 1) + np.cumsum(splitted_cells)
        cc[kept_cells] = np.iinfo(cc.dtype).max # just to generate exeception if used
        # face center (valid only for splitted faces)
        fc = (
            (self.nb_vertices + np.sum(splitted_cells) - 1)
            + np.cumsum(splitted_faces)
        )
        fc[kept_faces] = np.iinfo(fc.dtype).max # just to generate exeception if used
        return new_vertices, cc, fc
    
    def _new_cells(
        self, kept_cells, kept_faces, cell_centers=None, face_centers=None
    ):
        face_nodes = self.face_nodes
        cell_faces = self.cell_faces
        cell_nodes = self.cell_nodes
        if cell_centers is None:
            cell_centers = self._centers(self.cell_nodes)
        if face_centers is None:
            face_centers = self._centers(self.face_nodes)
        new_vertices, cc, fc = self._new_vertices(
            cell_centers, kept_cells, face_nodes, kept_faces, face_centers
        )
        new_cells = []
        for ci, kept in enumerate(kept_cells):
            if kept:
                new_cells.append([cell_nodes[ci]])
            else:
                parts = []
                cci = cc[ci]
                faces = cell_faces[ci]
                for fi in faces:
                    if kept_faces[fi]:
                        parts.append(list(face_nodes[fi]) + [cci,])
                    else:
                        fci = fc[fi]
                        nodes = face_nodes[fi]
                        for k in range(len(nodes)):
                            parts.append([nodes[k-1], nodes[k], fci, cci])
                new_cells.append(parts)
        assert len(new_cells)==self.nb_cells
        original_cell = np.fromiter(
            chain.from_iterable(
                [ci,]*len(parts) for ci, parts in enumerate(new_cells)
            ), dtype=MT.idt
        )
        new_cells = list(chain.from_iterable(new_cells))
        return new_vertices, new_cells, original_cell

    def as_tets(self, cell_centers=None):
        vertices, cells, original = self._new_cells(
            self.tetrahedron_cells(), self.triangle_faces(),
            cell_centers=cell_centers,
        )
        cells = np.array(cells, dtype=MT.idt)
        assert cells.ndim==2 and cells.shape[1]==4
        return MT.TetMesh.make(vertices, cells), original

    def as_hybrid_mesh(
        self, cell_centers=None, face_centers=None, convert_voxels=False
    ):
        vertices, cells, original = self._new_cells(
            self.tetrahedron_cells() | self.hexahedron_cells(),
            self.triangle_faces() | self.quadrangle_faces(),
            cell_centers=cell_centers, face_centers=face_centers
        )
        # CHECKME: the follwing creation might benefit from optimized routines
        mesh = MT.HybridMesh.Mesh()
        mesh_vertices = mesh.vertices
        Point = MT.Point
        for xyz in vertices:
            mesh_vertices.append(Point(xyz))
        Hexahedron = MT.Hexahedron
        hex_constructor = Hexahedron
        if convert_voxels:
            swap = np.arange(8)
            for i, j in [(2, 3), (6, 7)]:
                swap[i], swap[j] = swap[j], swap[i]
            hex_constructor = lambda nodes: MT.Hexahedron(MT.idarray(nodes)[swap])
        constructor = {
            4: MT.Tetrahedron,
            5: MT.Pyramid,
            6: MT.Wedge,
            8: hex_constructor,
         }
        cellnodes = mesh.connectivity.cells.nodes
        for cell in cells:
            cellnodes.append(constructor[len(cell)](MT.idarray(cell)))
        mesh.connectivity.update_from_cellnodes()
        return mesh, original
