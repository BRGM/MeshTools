# Dans un terminal :
# python petrel.py chemin_du_fichier_petrel
import numpy as np
import sys, re, os
import MeshTools as MT
import MeshTools.vtkwriters as vtkw
import MeshTools.PetrelMesh as PM
from MeshTools.RawMesh import RawMesh
# 2--3
# |  |
# 0--1

def read_file_argument(grdecl):
    kwargs = {}
    #rep = ('/home/jpvergnes/Travail/projets/CHARMS/'
    #        'petrel/maillage_CPG_albien_mofac/')
    #grdecl = rep + 'MOFAC_albien_eclips.GRDECL'
    dirname = os.path.dirname(grdecl)
    basename = os.path.basename(grdecl)
    newfile = '{0}_PYTHON'.format(grdecl)
    if not os.path.isfile(newfile):
        with open(grdecl) as f:
            for line in f:
                if '*' in line:
                    rewrite_file(grdecl, newfile)
                    grdecl = newfile
                    break
    else:
        grdecl = newfile
    def generator_include(f):
        line = next(f)
        while line:
            if 'INCLUDE' in line:
                yield re.findall("'(.*)'", next(f))[0]
            line = next(f)
    names = []
    with open(grdecl) as f:
        for name in generator_include(f):
            names.append(name)
    for name in names:
        oldfile = '{0}/{1}'.format(dirname, name)
        newfile = '{0}/{1}_PYTHON'.format(dirname, name)
        with open(oldfile) as f:
            for line in f:
                if 'Generated : Petrel' in line:
                    break
            name_variable = line.split()[0]
            if os.path.isfile(newfile):
                kwargs[name_variable] = newfile
            if not os.path.isfile(newfile):
                for line in f:
                    if '*' in line:
                        rewrite_file(oldfile, newfile)
                        kwargs[name_variable] = newfile
                        break
                else:
                    kwargs[name_variable] = oldfile
    return grdecl, kwargs

def rewrite_file(fichier, fichier_out):
    fout = open(fichier_out, 'w')
    with open(fichier) as f:
        for line in f:
            if '*' in line:
                data = []
                line = line.split()
                for value in line:
                    if value == '/':
                        data.append(value)
                    elif '*' in value:
                        nbr, value = value.split('*')
                        value = float(value)
                        nbr = int(nbr)
                        data.extend(nbr*[value])
                    else:
                        data.append(float(value))
                fout.write('{0}\n'.format(' '.join(list(map(str, data)))))
            elif line.strip() and not line.startswith('--'):
                fout.write(line)
    fout.close()

class PetrelGrid(object):
    def __init__(self, mainfile, **kwargs):
        zcornfile = kwargs.get('ZCORN', mainfile)
        coordfile = kwargs.get('COORD', mainfile)
        actnumfile = kwargs.get('ATNUM', mainfile)
        permxfile = kwargs.get('PERMX', mainfile)
        permyfile = kwargs.get('PERMY', mainfile)
        permzfile = kwargs.get('PERMZ', mainfile)
        with open(mainfile) as f:
            for line in f:
                if line.startswith('MAPAXES'):
                    self.mapaxes = np.asarray(next(f).strip().split(' ')[:-1],
                            'float')
                if line.startswith('SPECGRID'):
                    self.nx, self.ny, self.nz = np.asarray(
                            next(f).strip().split(' ')[:-3], 'int')
        coord = self._read_coord(coordfile)
        self._read_zcorn(zcornfile)
        # Vérifier que la correspondance des couches est cohérente (pas de trous)
        assert np.all(self.zcorn[:, :, 1:, :4] == self.zcorn[:, :, :-1, 4:]) 
        self.permx = self._read_grid(permxfile, 'PERMX')
        self.permy = self._read_grid(permyfile, 'PERMX')
        self.permz = self._read_grid(permzfile, 'PERMY')
        self.x = np.zeros(self.zcorn.shape)
        self.y = np.zeros(self.zcorn.shape)
        pilars = coord.reshape((self.nx+1, self.ny+1, 6), order='F')
        dxyz = pilars[:,:,3:] - pilars[:,:,:3] #bottom-top
        bad_pilars = dxyz[:,:,2] == 0
        # DZ nulles
        if np.any(bad_pilars):
            print('WARNING You have', np.sum(bad_pilars), 'bad pilars!')
        assert np.all(dxyz[:,:,2] >= 0) 
        # Vecteur directeur
        for i in [0, 4]: # top et bottom
            for k, pos in enumerate([
                    (slice(None,-1), slice(None,-1)),
                    (slice(1,None), slice(None,-1)),
                    (slice(None,-1), slice(1,None)),
                    (slice(1,None), slice(1,None)),
                    ]):
                for izc in range(self.nz):
                    dz = self.zcorn[:,:,izc,i+k] - pilars[pos+(2,)]
                    self.x[:,:,izc,i+k] = np.where(
                            dxyz[pos+(2,)] == 0,
                            pilars[pos+(0,)],
                            pilars[pos+(0,)] + dz*dxyz[pos+(0,)]/dxyz[pos+(2,)])
                    self.y[:,:,izc,i+k] = np.where(
                            dxyz[pos+(2,)] == 0,
                            pilars[pos+(1,)],
                            pilars[pos+(1,)] + dz*dxyz[pos+(1,)]/dxyz[pos+(2,)])
        assert np.all(self.x[:, :, 1:, :4] == self.x[:, :, :-1, 4:]) 
        assert np.all(self.y[:, :, 1:, :4] == self.y[:, :, :-1, 4:]) 
        # On enlève les cellules plates
        mask_cells = np.zeros((self.zcorn.shape[0],
            self.zcorn.shape[1], self.zcorn.shape[2], 4))
        for i in range(4):
            mask_cells[self.zcorn[:,:,:,i] == self.zcorn[:,:,:,i+4]] = 1
        mask_cells = np.sum(mask_cells, axis=-1)
        for i in range(1, 3):
            assert np.all(mask_cells != 1)
            assert np.all(mask_cells != 2)
            assert np.all(mask_cells != 3)
        mask_cells = np.where(mask_cells == 4, 0, 1)
        # On ajoute les cellules inactives au masque
        actnum = self._read_grid(actnumfile, 'ACTNUM')
        if actnum is not None:
            mask_cells = np.where(actnum == 0, 0, mask_cells)
        self.mask_cells = mask_cells
        mask_cells = np.repeat(mask_cells[:,:,:,np.newaxis], 8, axis=-1)
        self.x = np.where(mask_cells == 0, 9999., self.x)
        self.y = np.where(mask_cells == 0, 9999., self.y)
        self.zcorn = np.where(mask_cells == 0, 9999., self.zcorn)
        numcells = np.cumsum(self.mask_cells) - 1
        self.numcells = numcells.reshape(self.mask_cells.shape)
        del mask_cells

    def _read_grid(self, mainfile, field, **kwargs):
        with open(mainfile) as f:
            line = f.readline()
            while line:
                if line.startswith(field):
                    return np.fromfile(f, sep=' ',
                            count=self.nx*self.ny*self.nz, **kwargs).reshape(
                                    (self.nx, self.ny, self.nz), order='F')
                line = f.readline()

    def _read_coord(self, mainfile):
        with open(mainfile) as f:
            line = f.readline()
            while line:
                if line.replace('COORDSYS', '').startswith('COORD'):
                    coord = np.fromfile(f, sep=' ',
                        count=6*(self.nx+1)*(self.ny+1)).reshape((-1, 6))
                    return coord
                line = f.readline()

    def _read_zcorn(self, mainfile):
        self.zcorn = np.zeros((self.nx, self.ny, self.nz, 8))
        with open(mainfile) as f:
            line = f.readline()
            while line:
                if line.startswith('ZCORN'):
                    for k in range(self.nz):
                        for j in range(self.ny):
                            tmp = np.fromfile(f, sep=' ', count=2*self.nx).reshape(
                                                    (self.nx, 2))
                            self.zcorn[:, j, k, 0:2] = tmp
                            tmp = np.fromfile(f, sep=' ', count=2*self.nx).reshape(
                                                    (self.nx, 2))
                            self.zcorn[:, j, k, 2:4] = tmp
                        for j in range(self.ny):
                            tmp = np.fromfile(f, sep=' ', count=2*self.nx).reshape(
                                                    (self.nx, 2))
                            self.zcorn[:, j, k, 4:6] = tmp
                            tmp = np.fromfile(f, sep=' ', count=2*self.nx).reshape(
                                                    (self.nx, 2))
                            self.zcorn[:, j, k, 6:8] = tmp
                        print("Variable ZCORN de la couche {0} lue".format(k + 1))
                    break
                line = f.readline()

    def process(self):
        vertices = []
        ids = []
        new_ids = np.zeros(8, dtype=np.long)
        xpil = np.ones((self.nx+1, self.ny+1, self.nz+1, 8))*9999
        ypil = np.ones((self.nx+1, self.ny+1, self.nz+1, 8))*9999
        zpil = np.ones((self.nx+1, self.ny+1, self.nz+1, 8))*9999
        for u, v in zip([xpil, ypil, zpil], [self.x, self.y, self.zcorn]):
            u[:-1, :-1, :-1, 0] = v[:, :, :, 0]
            u[1:, :-1, :-1, 1] = v[:, :, :, 1]
            u[:-1, 1:, :-1, 2] = v[:, :, :, 2]
            u[1:, 1:, :-1, 3] = v[:, :, :, 3]
            u[:-1, :-1, 1:, 4] = v[:, :, :, 4]
            u[1:, :-1, 1:, 5] = v[:, :, :, 5]
            u[:-1, 1:, 1:, 6] = v[:, :, :, 6]
            u[1:, 1:, 1:, 7] = v[:, :, :, 7]
        xpil.shape = (-1, 8)
        ypil.shape = (-1, 8)
        zpil.shape = (-1, 8)
        # Boucle sur les piliers
        for zci, zc in enumerate(zpil):
            for j in range(len(zc)): # len(z) = 8 
                for k in range(j):
                    if zc[k]==zc[j]:
                        new_ids[j] = new_ids[k]
                        break
                else:
                # Si z = 9999., point masqué on ne le rajoute pas
                    if zc[j] != 9999.:
                        X = (xpil[zci][j], ypil[zci][j], zc[j])
                        if X not in vertices:
                            new_ids[j] = len(vertices)
                            X = (xpil[zci][j], ypil[zci][j], zc[j])
                            vertices.append(X)
                        else:
                            indice = vertices.index(X)
                            new_ids[j] = indice
            ids.append(np.copy(new_ids)) # la copie est importante ici
        # normalement ici on a le tableaux des vertices et 'yapluka' construire tous les hexagones
        vertices = np.array(vertices)
        corner_ids = np.array(ids)
        corner_ids = corner_ids.reshape((self.nx + 1, self.ny + 1, self.nz+1, 8))
        # jolie boucle imbriquée ???
        hexagons = []
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if self.mask_cells[i,j,k] == 1:
                        hexagon = (
                            # Je ne sais pas dans quel sens on tourne !!!!
                            corner_ids[i, j, k, 0],
                            corner_ids[i+1, j, k, 1],
                            corner_ids[i+1, j+1, k, 3],
                            corner_ids[i, j+1, k, 2],
                            corner_ids[i, j, k+1, 4],
                            corner_ids[i+1, j, k+1,  5],
                            corner_ids[i+1, j+1, k+1, 7],
                            corner_ids[i, j+1, k+1, 6],
                        )
                        hexagons.append(hexagon)
        return vertices, MT.idarray(hexagons)
    
    def get_perm(self):
        permx, permy, permz = [], [], []
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if self.mask_cells[i,j,k] == 1:
                        if self.permx is not None:
                            permx.append(self.permx[i,j,k])
                        if self.permy is not None:
                            permy.append(self.permy[i,j,k])
                        if self.permz is not None:
                            permz.append(self.permz[i,j,k])
        return permx, permy, permz
        

    def get_faces(self, cells):
        faces = []
        indices = [
                [0, 1, 2, 3], # Face top
                [4, 5, 6, 7], # Face bottom
                [0, 4, 5, 1], # Face back
                [3, 7, 6, 2], # Face front
                [0, 4, 7, 3], # Face left
                [1, 5, 6, 2]] # Face right
        for i, indice in enumerate(indices):
            faces.extend(np.array(cells)[:, indice].tolist())
        faces_sort = np.sort(faces, axis=1)
        newfaces, unique_indices, face_id = np.unique(faces_sort,
                axis=0, return_inverse=True, return_index=True)
        newfaces_nodes = np.array(faces)[unique_indices]
        newcells_faces = face_id.reshape(
                    (len(cells), len(indices)), order='F')
        return newfaces_nodes, newcells_faces

    def get_new_faces(self, pvertices, cells_faces, face_nodes):
        self.new_faces_nodes = face_nodes.tolist()
        self.new_cells_faces = cells_faces.tolist()
        self.pvertices = pvertices
        # en X
        self.corrections_faces(cells_faces, face_nodes, [1, 0])
        # en Y
        self.corrections_faces(cells_faces, face_nodes, [0, 1])
        return self.pvertices, self.new_faces_nodes, self.new_cells_faces

    def corrections_faces(self, cells_faces, face_nodes, shift):
        maxnode = len(self.pvertices)
        # Pour le moment, uniquement les failles selon x
        for ix in range(self.nx - shift[0]):
            for jy in range(self.ny - shift[1]):
                # Numéros de cellules colonne 1 (gauche) et 2 (droite)
                numcellsl = self.numcells[ix,jy]
                numcellsr = self.numcells[ix+shift[0],jy+shift[1]]
                # Numéros de faces colonne 1 (gauche)
                cellsfacel = cells_faces[numcellsl][:,-1-shift[1]*2]
                # Numéros de faces colonne 2 (droite)
                cellsfacer = cells_faces[numcellsr][:,-2-shift[1]*2]
                # Numéros des noeuds piliers back front
                backr = face_nodes[cellsfacer][:,:2]
                backl = face_nodes[cellsfacel][:,:2]
                diff = self.pvertices[backr] - self.pvertices[backl]
                if np.nonzero(diff)[0].shape[0] == 0:
                    continue
                frontr = face_nodes[cellsfacer][:,2:]
                frontl = face_nodes[cellsfacel][:,2:]
                nodesfront = np.unique(np.concatenate([frontr, frontl]))
                nodesback = np.unique(np.concatenate([backr, backl]))
                ordrepilierfront = np.argsort(self.pvertices[nodesfront][:,2])
                ordrepilierback = np.argsort(self.pvertices[nodesback][:,2])
                # Vertices des piliers plan yz
                numberback = np.array(nodesback)[ordrepilierback]
                numberfront = np.array(nodesfront)[ordrepilierfront]
                verticesback = self.pvertices[nodesback][ordrepilierback]
                verticesfront = self.pvertices[nodesfront][ordrepilierfront]
                numbers = np.concatenate([numberback, numberfront])
                seg_nback = np.arange(len(numberback))
                seg_nfront = np.arange(len(numberback), len(numbers))
                segmentsl = face_nodes[cellsfacel][:,[0,3,1,2]].reshape(
                        len(cellsfacel)*2, 2)
                segmentsl = np.unique(segmentsl, axis=0)
                segmentsr = face_nodes[cellsfacer][:,[0,3,1,2]].reshape(
                        len(cellsfacer)*2, 2)
                segmentsr = np.unique(segmentsr, axis=0)
                lsegments = [[], []]
                for iseg, segs in enumerate([segmentsl, segmentsr]):
                    for segment in segs:
                        i = np.where(numberback == segment[0])[0][0]
                        j = np.where(numberfront == segment[1])[0][0]
                        new_segment = [seg_nback[i], seg_nfront[j]]
                        if new_segment not in lsegments[iseg]:
                            lsegments[iseg].append(new_segment)
                tsegments = np.concatenate(lsegments)
                tsegments = np.array(tsegments, dtype=np.int32)
                # Normalisation sur les nouveaux repèries
                def pilar_referential(pts):
                    z = pts[:, 2]
                    kmin = np.argmin(z)
                    kmax = np.argmax(z)
                    e = pts[kmax]-pts[kmin]
                    return pts[kmin], e
                Oback, uback = pilar_referential(verticesback)
                Ofront, ufront = pilar_referential(verticesfront)
                nvback = np.zeros((verticesback.shape[0], 2))
                nvfront = np.ones((verticesfront.shape[0], 2))
                def new_coord(pilarpts, O, e): 
                    res = np.sum((pilarpts-O)*e, axis=1)
                    e2 = np.sum(e*e)
                    assert e2>0
                    res/= e2
                    return res
                nvback[:, 1] = new_coord(verticesback, Oback, uback)
                nvfront[:, 1] = new_coord(verticesfront, Ofront, ufront)
                nv = np.vstack([nvback, nvfront])
                uv, triangles, components, faces = PM.mesh(
                        nv.astype('float64'), tsegments)
                nv2 = np.reshape(uv[:,0], (-1, 1)) * ( 
                        Ofront + np.tensordot(uv[:,1],  ufront, axes=0)
                )
                nv2+= np.reshape(1-uv[:,0], (-1, 1)) * (
                        Oback + np.tensordot(uv[:,1], uback, axes=0)
                )
                # 1. Ajout des nouveaux noeuds : interpolation bilinéaire
#                if uv.shape[0] > nv.shape[0]:
#                    newvertices = []
#                    newpoints = uv[nv.shape[0]:,:]
#                    for i in range(3):
#                        # f du point (1,0)
#                        deltaf = verticesfront[0, i] - verticesfront[-1, i]
#                        deltav = nvfront[0, 2] - nvfront[-1, 2]
#                        r = deltaf / deltav
#                        flr = verticesfront[-1, i] - nvfront[-1, 2] * r
#                        # f du point (1,1)
#                        fur = verticesfront[0, i] + (1-nvfront[0, 2]) * r
#                        # f du point (0,0)
#                        deltaf = verticesback[0, i] - verticesback[-1, i]
#                        deltav = nvback[0, 2] - nvback[-1, 2]
#                        r = deltaf / deltav
#                        fll = verticesback[-1, i] - nvback[-1, 2] * r
#                        # f du point (0,1)
#                        ful = verticesback[0, i] + (1-nvback[0, 2]) * r
#                        coef1  = (flr + newpoints[:,1]*(fur-flr))
#                        coef2  = (fll + newpoints[:,1]*(ful-fll))
#                        newvertices.append(newpoints[:,0]*coef1
#                                + (1-newpoints[:,0])*coef2)
#                    newvertices = np.array(newvertices).T
#                    #newvertices = np.round(np.array(newvertices).T, 2)
#                    self.pvertices = np.concatenate([self.pvertices, newvertices])
                assert np.linalg.norm(
                        nv2[:nv.shape[0]]
                        -np.vstack([verticesback, verticesfront]), 
                    axis=1).max()<1E-5
                newvertices = nv2[nv.shape[0]:]
                self.pvertices = np.concatenate([self.pvertices, newvertices])
                # 2. On retrouve le numéro des cellules correspondant aux nouvelles faces
                tcenters = np.vstack([uv[triangle].mean(axis=0) for triangle in triangles])
                numcells = [numcellsl, numcellsr]
                indices_cells = [[], []] # [ left, right ]
                components_cells = [[], []] # [ left, right ]
                for i, segment in enumerate(lsegments):
                    segment = np.array(segment)
                    pback = uv[segment][:, 0] # y
                    pfront = uv[segment][:, 1] # z
                    a = (pfront[:, 1] - pback[:,1])/(pfront[:, 0] - pback[:, 0])
                    b = pfront[:, 1] - a * pfront[:, 0]
                    for component, center in zip(components, tcenters):
                        points = a * center[0] + b - center[1]
                        #fi = np.sort(np.argsort(np.abs(points))[:2])[0]
                        # On passe sur tous les points des segments
                        for fi, point in enumerate(points):
                            if point > 0:
                                if fi == 0:
                                    break
                                else:
                                    indices_cells[i].append(numcells[i][fi-1])
                                    components_cells[i].append(component)
                                    break
                # 3. Description des cellules par leurs nouvelles faces
                noeuds_ajout = []
                for i, cellsface in enumerate([cellsfacel, cellsfacer]): # left, right
                    com = np.array(components_cells[i])
                    ind = np.array(indices_cells[i])
                    ncells = np.unique(ind)
                    num_faces = []
                    for jcell, j in enumerate(range(ncells.min(), ncells.max()+1)):
                        cell = self.new_cells_faces[j]
                        # Faces décrite par leurs noeuds
                        # correspondant aux numéros de cellules j
                        # faces : numéro de noeud de la triangulation
                        faces_j = np.array(faces)[np.unique(com[ind == j])].tolist()
                        # Boucle sur ces faces. Vérification si elles sont déjà
                        # présentes dans face_nodes
                        modif = False
                        for face in faces_j:
                            newface = []
                            for noeud_tri in face:
                                # Si noeud_tri inférieur à nv, noeud déjà existant
                                if noeud_tri < nv.shape[0]:
                                    newface.append(numbers[noeud_tri])
                                # Sinon nouveau noeud, et on l'ajoute
                                else:
                                    newface.append((maxnode + noeud_tri - nv.shape[0] + 1))
                                    noeuds_ajout.append(noeud_tri - nv.shape[0] + 1)
                            newface = np.array(newface)
                            face_sort = np.sort(newface).tolist()
                            if face_sort not in np.sort(face_nodes[cellsface]).tolist():
                                # Si pas dans face_nodes_lr
                                # On remplace dans face_nodes, mais on garde le numero de face
                                if not modif:
                                    self.new_faces_nodes[cellsface[jcell]] = newface.tolist()
                                    modif = True
                                # On rajoute une face, donc on rajoute aussi dans la
                                # description des cellules par leur face
                                else:
                                    self.new_faces_nodes.append(newface.tolist())
                                    cell.append(len(self.new_faces_nodes)-1)
                        # Mise à jour des nouvelles cellules
                        if modif:
                            self.new_cells_faces[j] = cell
                    #if 707 in numcells[1]:
                    #    import ipdb; ipdb.set_trace()
                if noeuds_ajout:
                    nmax = np.max(noeuds_ajout)
                    assert(nmax == len(np.unique(noeuds_ajout)))
                    assert(nmax + maxnode == len(self.pvertices) - 1)
                maxnode = len(self.pvertices) - 1

if __name__ == "__main__":
    grdecl = sys.argv[1]
    grdecl, kwargs = read_file_argument(grdecl)
    pgrid = PetrelGrid(grdecl, **kwargs)
    pvertices, phexagons = pgrid.process()
    permx, permy, permz = pgrid.get_perm()
    #face_nodes, cells_faces = pgrid.get_faces(phexagons)
    #pvertices2, face_nodes2, cells_faces2 = pgrid.get_new_faces(
    #        pvertices, cells_faces, face_nodes)
    #mesh = RawMesh(
    #    vertices=pvertices2,
    #    face_nodes=face_nodes2,
    #    cell_faces=cells_faces2
    #)
    #cell_property = np.array([np.sqrt(2), np.pi]) # 2 cells
    #tetmesh, original_cell = mesh.as_tets()
    #MT.to_vtu(
    #    tetmesh, 'cells_as_tets',
    #    celldata = {
    #        'original_cell': original_cell,
    #        #'magic_numbers': cell_property[original_cell],
    #    },
    #)
    mesh = MT.HybridMesh.Mesh()
    vertices = mesh.vertices
    for P in pvertices:
        vertices.append(MT.Point(P))
    cellnodes = mesh.connectivity.cells.nodes
    for elt in phexagons:
        cellnodes.append(MT.Hexahedron(elt))
    mesh.connectivity.update_from_cellnodes()
    offsets, cellsnodes = mesh.cells_nodes_as_COC()
    vtkw.write_vtu(vtkw.vtu_doc_from_COC(mesh.vertices_array(), 
                np.array(offsets[1:], copy=False), # no first zero offset for vtk 
                np.array(cellsnodes, copy=False), mesh.cells_vtk_ids()),
                'petrel2.vtu') 
