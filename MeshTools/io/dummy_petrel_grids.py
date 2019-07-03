from itertools import product
import numpy as np

from petrel import PetrelGrid

def petrel_unit_cube(dtype=np.double):
    """Build a unit cube defined by its 8 nodes
    in the order expected by Petrel"""
    nodes = np.zeros((8,3), dtype=dtype)
    nodes[1::2, 0] = 1 # right face  x = 1
    nodes[ 2:4, 1] = 1 # back face   y = 1
    nodes[  6:, 1] = 1 # back face   y = 1
    nodes[  4:, 2] = 1 # bottom face z = 1 (3rd coord is depth!)
    return nodes

def add_buffer(hexaedra):
    ncx, ncy, ncz, _, dim = hexaedra.shape
    assert ncx>0 and ncy>0 and ncz>0 and _==8 and dim==3
    buffered = np.empty((ncx+2, ncy+2, ncz, 8, 3), dtype=hexaedra.dtype)
    buffered[1:(ncx+1), 1:(ncy+1), ...] = hexaedra
    buffered[0, 1:(ncy+1), :, 1::2] = hexaedra[0, :, :, 0::2]
    buffered[ncx+1, 1:(ncy+1), :, 0::2] = hexaedra[-1, :, :, 1::2]
    buffered[1:(ncx+1), 0, :, (2, 3, 6, 7)] = hexaedra[:, 0, :, (0, 1, 4, 5)]
    buffered[1:(ncx+1), ncy+1, :, (0, 1, 4, 5)] = hexaedra[:, -1, :, (2, 3, 6, 7)]
    buffered[ 0,  0, :, (3, 7)] = hexaedra[ 0,  0, :, (0, 4)]
    buffered[ 0, -1, :, (1, 5)] = hexaedra[ 0, -1, :, (2, 6)]
    buffered[-1,  0, :, (2, 6)] = hexaedra[-1,  0, :, (1, 5)]
    buffered[-1, -1, :, (0, 4)] = hexaedra[-1, -1, :, (3, 7)]
    return buffered

def add_mask_buffer(hexaedra):
    buffered = np.ma.array(add_buffer(hexaedra))
    mask = np.zeros(buffered.shape, dtype=np.bool)
    mask[ 0,  :, :, (0, 2, 4, 6), :] = True
    mask[-1,  :, :, (1, 3, 5, 7), :] = True
    mask[ :,  0, :, (0, 1, 4, 5), :] = True
    mask[ :, -1, :, (2, 3, 6, 7), :] = True
    buffered.mask = mask
    return buffered

def collect_pilars_nodes(hexaedra):
    ncx, ncy, ncz, _, dim = hexaedra.shape
    assert ncx>0 and ncy>0 and ncz>0 and _==8 and dim==3
    buffered = add_buffer(hexaedra)
    pilars_nodes = np.concatenate(
        (
            buffered[:-1, :-1, :, (3, 7), :],
            buffered[:-1,  1:, :, (1, 5), :],
            buffered[ 1:, :-1, :, (2, 6), :],
            buffered[ 1:,  1:, :, (0, 4), :],
        ), axis=3,
    )
    assert pilars_nodes.shape[:2]==(ncx+1, ncy+1)
    assert np.all(pilars_nodes[..., 1::2, 2]>=pilars_nodes[..., 0::2, 2])
    return pilars_nodes
    
def pilars(hexaedra):
    pilars_nodes = collect_pilars_nodes(hexaedra)
    i, j = np.indices(pilars_nodes.shape[:2])
    top = pilars_nodes[:, :, 0, 0::2]
    bottom = pilars_nodes[:, :, -1, 1::2]
    pilars = np.concatenate(
        (
            top[i, j, np.argmin(top[..., 2], axis=2), None], # z is depth
            bottom[i, j, np.argmax(bottom[..., 2], axis=2), None], # z is depth
        ), axis=-1,
    )
    assert pilars.shape[2]==1
    return pilars[..., 0, :]

def four_cells_stairs(vertical_translation=0.2, cube=None):
    if cube is None:
        cube = petrel_unit_cube()
    t = lambda x, y, z: np.array((x, y, z), dtype=cube.dtype)
    cells = np.array([
        cube + t(i, j, k*vertical_translation)
        for k, (i, j) in enumerate(product((0,1), (0,1)))
    ])
    cells.shape = (2, 2, 1, 8, 3)
    return cells

def grid_of_heaxaedra(shape, cube=None):
    assert np.all(np.array(shape)>0) and len(shape)==3
    if cube is None:
        cube = petrel_unit_cube()
    t = lambda x, y, z: np.array((x, y, z), dtype=cube.dtype)
    cells = np.array([
        cube + t(i, j, k) for i, j, k in product(*(range(n) for n in shape))
    ])
    cells.shape = shape + (8, 3)
    return cells

def faulted_ramp(shape, begin=0.5, voffset=1.):
    nx, ny, nz = shape
    assert nx>0 and ny>0 and nz>0 and ny%2==0
    cells = grid_of_heaxaedra(shape)
    L = float(nx)
    half = cells[:, :(ny//2), ...]
    x = half[..., 0]
    x0 = begin * L
    offset = np.zeros(half.shape[:-1], dtype=np.double)
    where = x>x0
    offset[where] = voffset * (1 - np.cos((x[where]-x0)*(np.pi/L)))
    cells[:, :(ny//2), ..., 2]+= offset
    return cells

def various_dirty_checks_to_be_cleaned():
    cube = petrel_unit_cube()
    t = lambda x, y, z: np.array((x, y, z), dtype=cube.dtype)
    def vertical_pilars(nodes):
        nx, ny = nodes.shape[:2]
        test = np.zeros((nx, ny), dtype=np.bool)
        for i, j in product(range(nx), range(ny)):
            test[i, j] = np.all([
                    nodes[i, j, :, 0, :2]==nodes[i, j, :, k, :2]
                ] for k in range(8)
            )
        return np.all(test)
    cube_grid = cube[None, None, None, ...]
    # print(add_buffer(cube))
    # print(collect_pilars_nodes(cube_grid))
    two_cubes = np.array([cube, cube+t(1, 0, 0.5)], dtype=cube.dtype)
    two_cubes_grid = two_cubes[:, None, None, ...]
    pilars_nodes = collect_pilars_nodes(two_cubes_grid)
    assert vertical_pilars(pilars_nodes), 'pilars should be vertical'
    two_cubes = np.array([cube, cube+t(0, 1, 0.5)], dtype=cube.dtype)
    two_cubes_grid = two_cubes[None, :, None, ...]
    pilars_nodes = collect_pilars_nodes(two_cubes_grid)
    # print(two_cubes_grid.shape)
    # print()
    buffer = add_mask_buffer(two_cubes_grid)
    nx, ny = buffer.shape[:2]
    # for i in range(nx):
        # for j in range(ny):
            # print('- (%d, %d) %s' % (i, j, '-'*20))
            # print(buffer[i, j, 0, :4])
            # print()
    # print(
        # 'buffer[:-1, :-1, :, 3]\n', buffer[:-1, :-1, :, 3, :], '\n\n',
        # 'buffer[:-1,  1:, :, 1]\n', buffer[:-1,  1:, :, 1, :], '\n\n',
        # 'buffer[ 1:, :-1, :, 2]\n', buffer[ 1:, :-1, :, 2, :], '\n\n',
        # 'buffer[ 1:,  1:, :, 0]\n', buffer[ 1:,  1:, :, 0, :], '\n\n',
    # )
    # for i in range(nx-1):
        # for j in range(ny-1):
            # print(
                # buffer[:-1, :-1, :, 3, :][i, j, ...], '\n',
                # buffer[:-1,  1:, :, 1, :][i, j, ...], '\n',
                # buffer[ 1:, :-1, :, 2, :][i, j, ...], '\n',
                # buffer[ 1:,  1:, :, 0, :][i, j, ...], '\n',
            # )
            # print()
    # print('concatenate')
    test = np.concatenate(
            (
                buffer[:-1, :-1, :, (3, 7), :],
                buffer[:-1,  1:, :, (1, 5), :],
                buffer[ 1:, :-1, :, (2, 6), :],
                buffer[ 1:,  1:, :, (0, 4), :],
            ), axis=3,
        )
    assert np.all(test==collect_pilars_nodes(two_cubes_grid))
    # print(test)
    # print(pilars_nodes)
    assert vertical_pilars(pilars_nodes), 'pilars should be vertical'
    stairs = four_cells_stairs()
    buffered = add_buffer(stairs)
    pilars_nodes = collect_pilars_nodes(stairs)
    assert vertical_pilars(pilars_nodes), 'pilars should be vertical'
    # print('*'*20)
    # print(stairs)
    # print(pilars_nodes)
    # print(pilars(stairs))
    cells = grid_of_heaxaedra((4, 3, 2))
    # print(cells[0,0,0])
    # print(cells[0,0,1])

def depth_to_elevation(pts, zref=None, zmap=None, copy=False):
    if zmap is not None:
        assert zref is None
        assert False, 'not implemented'
    elif zref is None:
        zref = 0
    assert pts.shape[-1]==3
    res = np.array(pts, copy=copy)
    res[..., 2] = zref - res[..., 2]
    return res

def minimum_distance(pts):
    n, dim = pts.shape
    assert dim==3
    return np.min([
        np.linalg.norm(pts[j]-pts[i])
        for i, j in product(range(n), range(n)) if i<j
    ])

if __name__=='__main__':
    various_dirty_checks_to_be_cleaned()
    import MeshTools as MT
    from MeshTools.RawMesh import RawMesh
    def build_and_dump(hexaedra, name):
        print('-'*50)
        print('processing', name)
        pil = pilars(hexaedra)
        grid = PetrelGrid.build_from_arrays(
            pil[..., :3], pil[..., 3:], hexaedra[..., 2]
        )
        print('pilars shape', pil.shape)
        print('zcorn shape', hexaedra[..., 2].shape)
        hexa = grid.process()
        print('minimum distance', minimum_distance(grid.pvertices))
        MT.to_vtu(
            MT.HexMesh.make(depth_to_elevation(grid.pvertices), hexa), 
            name,
        )
        grid.set_faces(hexa)
        mesh = RawMesh(
            vertices=grid.pvertices,
            face_nodes=grid.new_faces_nodes,
            cell_faces=grid.cells_faces
        )
        print(
            'mesh with', mesh.nb_cells, 'cells', mesh.nb_faces, 'faces',
            mesh.nb_vertices, 'vertices'
        )
        print('number of tetrahedra:', np.sum(mesh.tetrahedron_cells()))
        print('number of hexahedra:', np.sum(mesh.hexahedron_cells()))
        # raw_mesh, original_cell = mesh.as_tets()
        raw_mesh, original_cell = mesh.as_hybrid_mesh(convert_voxels=True)
        MT.to_vtu(
            raw_mesh, name + '_splitted',
            celldata = {'original_cell': original_cell},
        )
    # --------------------------------------------------------------------------
    build_and_dump(grid_of_heaxaedra((4, 3, 2)), 'sugar_box')
    build_and_dump(four_cells_stairs(), 'stairs')
    build_and_dump(faulted_ramp((8, 2, 1), begin=0.33), 'ramp')
    build_and_dump(faulted_ramp((8, 4, 3), begin=0.33), 'thicker_ramp')
    
