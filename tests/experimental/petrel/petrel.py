import numpy as np
# import pylab as plt
import MeshTools as MT
import MeshTools.vtkwriters as vtkw
import MeshTools.PetrelMesh as PM
# 2--3
# |  |
# 0--1

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
        zcornfile = kwargs.get('zcornfile', mainfile)
        coordfile = kwargs.get('coordfile', mainfile)
        actnumfile = kwargs.get('actnumfile', mainfile)
        propfaciesfile = kwargs.get('propfaciesfile', mainfile)
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
        self.propfacies = self._read_grid(propfaciesfile, 'FACIES')
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
        maxnode = np.max(self.new_faces_nodes)
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
                ordrepilierback = np.argsort(self.pvertices[nodesback][:,2])
                ordrepilierfront = np.argsort(self.pvertices[nodesfront][:,2])
                minval = np.min(np.concatenate([nodesback, nodesfront]))
                # Vertices des piliers plan yz
                numberback = np.array(nodesback)[ordrepilierback]
                numberfront = np.array(nodesfront)[ordrepilierfront]
                verticesback = self.pvertices[nodesback][ordrepilierback]
                verticesfront = self.pvertices[nodesfront][ordrepilierfront]
                numbers = np.concatenate([numberback, numberfront])
                segments1 = face_nodes[cellsfacel][:,[0,3,1,2]].reshape(
                        len(cellsfacel)*2, 2)
                segments1 = np.unique(segments1, axis=0)
                segments2 = face_nodes[cellsfacer][:,[0,3,1,2]].reshape(
                        len(cellsfacer)*2, 2)
                segments2 = np.unique(segments2, axis=0)
                lsegments = [[], []]
                for iseg, segs in enumerate([segments1, segments2]):
                    for segment in segs:
                        i = np.where(numberback == segment[0])[0][0]
                        j = np.where(numberfront == segment[1])[0][0]
                        lsegments[iseg].append([i, len(nodesback) + j])
                tsegments = np.concatenate(lsegments)
                tsegments = np.unique(tsegments, axis=0)
                tsegments = np.array(tsegments, dtype=np.int32)
                # Normalisation sur les nouveaux repères
                nvback = np.zeros(verticesback.shape)
                nvfront = np.ones(verticesfront.shape)
                zall = np.concatenate([verticesback[:,2], verticesfront[:,2]])
                for nv, v in zip([nvback, nvfront],
                        [verticesback, verticesfront]):
                    z = v[:, 2]
                    z = (z-zall.min())/(zall.max()-zall.min())
                    nv[:, 2] = z
                nv = np.concatenate([nvback, nvfront])
                print(nv)
                #nv = np.round(nv, 5)
                # Triangulation
                #if [65, 67] in backr:
                #    import ipdb; ipdb.set_trace()
                uv, triangles, components, faces = PM.mesh(
                        nv[:, 1:].astype('float64'), tsegments)
                #uv = np.round(uv, 5)
                tcenters = np.vstack([uv[triangle].mean(axis=0) for triangle in triangles])
                # 1. Ajout des nouveaux noeuds : interpolation bilinéaire
                if uv.shape[0] > nv.shape[0]:
                    newvertices = []
                    newpoints = uv[nv.shape[0]:,:]
                    for i in range(3):
                        coef1  = (verticesfront[0,i] +
                                newpoints[:,1]*(verticesfront[-1,i]-verticesfront[0,i]))
                        coef2  = (verticesback[0,i] +
                                newpoints[:,1]*(verticesback[-1,i]-verticesback[0,i]))
                        newvertices.append(newpoints[:,0]*coef1 + (1-newpoints[:,0])*coef2)
                    newvertices = np.round(np.array(newvertices).T, 2)
                    self.newvertices = np.concatenate([self.pvertices, newvertices])
                # 2. On retrouve le numéro des cellules correspondant aux nouvelles faces
                tcenters = np.vstack([uv[triangle].mean(axis=0) for triangle in triangles])
                numcells = [numcellsr, numcellsl]
                indices_cells = [[], []] # [ right, left ]
                components_cells = [[], []] # [ right, left ]
                for i, segment in enumerate(lsegments):
                    segment = np.array(segment)
                    p1 = uv[segment][:, 0] # y
                    p2 = uv[segment][:, 1] # z
                    a = (p2[:, 1] - p1[:,1])/(p2[:, 0] - p1[:, 0])
                    b = p1[:, 1] - a * p1[:, 0]
                    for component, center in zip(components, tcenters):
                        points = a * center[0] + b - center[1]
                        find = False
                        for fi, point in enumerate(points):
                            if point > 0:
                                if fi == 0:
                                    break
                                else:
                                    indices_cells[i].append(numcells[i][fi-1])
                                    components_cells[i].append(component)
                                    break
                # 3. Description des cellules par leurs nouvelles faces
                for i, cellsface in enumerate([cellsfacer, cellsfacel]): # right, left
                    com = np.array(components_cells[i])
                    ind = np.array(indices_cells[i])
                    ncells = np.unique(ind)
                    num_faces = []
                    for jcell, j in enumerate(range(ncells.min(), ncells.max()+1)):
                        cell = self.new_cells_faces[j]
                        # Faces décrite par leurs noeuds
                        # correspondant aux numéros de cellules j
                        faces_j = np.array(faces)[np.unique(com[ind == j])].tolist()
                        # Boucle sur ces faces. Vérification si elles sont déjà
                        # présentes dans face_nodes
                        modif = False
                        for face in faces_j:
                            newface = []
                            for fa in face:
                                if fa < nv.shape[0]:
                                    newface.append(numbers[fa])
                                else:
                                    newface.append(fa + maxnode)
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
                maxnode = len(self.new_faces_nodes)

if __name__ == "__main__":
    rep = ('/home/jpvergnes/Travail/projets/CHARMS/'
            'petrel/maillage_CPG_albien_mofac/')
    grdecl = rep + 'MOFAC_albien_eclips.GRDECL'
    coordfile= rep + 'MOFAC_albien_eclips_COORD.GRDECL'
    #zcornfile = rep + 'MOFAC_albien_eclips_ZCORN.GRDECL'
    zcornfile_new = rep + 'MOFAC_albien_eclips_ZCORN_new.GRDECL'
    #rewrite_file(zcornfile, zcornfile_new)
    #actnumfile = rep + 'MOFAC_albien_eclips_ACTNUM.GRDECL'
    actnumfile_new = rep + 'MOFAC_albien_eclips_ACTNUM_new.GRDECL'
    #rewrite_file(actnumfile, actnumfile_new)
    #propfaciesfile = rep + 'MOFAC_albien_eclips_PROP_FACIES_NEW.GRDECL'
    propfaciesfile_new = rep + 'MOFAC_albien_eclips_PROP_FACIES_NEW_new.GRDECL'
    #rewrite_file(propfaciesfile, propfaciesfile_new)
    rep = ('/home/jpvergnes/Travail/projets/CHARMS/'
            'petrel/cas_test_simples_storengy/SimpleGridCases/')
    grdecl = rep + 'SIMPLE_GRID3.GRDECL'
    #rep = ('/home/jpvergnes/Travail/projets/CHARMS/'
    #        'petrel/cas_test_simples_storengy/')
    #grdecl = rep + 'Modele_CPG_Znonparallele_Actnum_new.GRDECL'
    #grdecl = rep + 'Modele_CPG_Znonparallele_new.GRDECL'
    #rewrite_file(grdecl, grdecl_new)
    pgrid = PetrelGrid(grdecl)#,
            #coordfile=coordfile,
            #zcornfile=zcornfile_new,
            #propfaciesfile=propfaciesfile_new,
            #actnumfile=actnumfile_new)
    pvertices, phexagons = pgrid.process()
    face_nodes, cells_faces = pgrid.get_faces(phexagons)
    pvertices, face_nodes, cells_faces = pgrid.get_new_faces(
            pvertices, cells_faces, face_nodes)
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

# Liste de liste de noeuds numérotés [[1,2,3,...9]]
# Liste des numéros de face [[1,2,3..4], [1,2,3...]]
# Tableau de noeuds
# Décrire les cellules par leurs noeuds (liste de listes)
# Décrire les faces par leurs noeuds (liste de listes)
# Décrire les cellules par leurs faces (liste de listes)
# Classer entre 0 et 1 les z sur chaque pilier
# Module shapely
# Outil de triangulation ??


# Face de faille : devant derrière, considérer que les points devant/derrière pour les 2 piliers
# Division des segments/Calcul de toutes les intersections : plein de segments (noeuds1, noeud2)
# Mailleur triangulaire : rajoute pas de points et triangule moi les faces
# Calcul composantes connexes -> transformer en polygones -> liste de polygones
# Attribuer les polygones aux faces devant derrière !! récupérer les z des cellules à côté
# Object polygone : côtés + barycentre

# Segments + liste de noeuds


# exemple BP, point de départ : vertical_column.py
