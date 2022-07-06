import glob

NUM_POINTS = 8192
shapenet_path = '/opt/data3/datasets/ShapeNetCore.v2/' # https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip
out_name = '/opt/data3/yuxumin/ShapeNet8k/'
import os,sys
import numpy as np
import trimesh

import json

def sample_triangle(v, n=None):
    if hasattr(n, 'dtype'):
        n = np.asscalar(n)
    if n is None:
        size = v.shape[:-2] + (2,)
    elif isinstance(n, int):
        size = (n, 2)
    elif isinstance(n, tuple):
        size = n + (2,)
    elif isinstance(n, list):
        size = tuple(n) + (2,)
    else:
        raise TypeError('n must be int, tuple or list, got %s' % str(n))
    assert(v.shape[-2] == 2)
    a = np.random.uniform(size=size)
    mask = np.sum(a, axis=-1) > 1
    a[mask] *= -1
    a[mask] += 1
    a = np.expand_dims(a, axis=-1)
    return np.sum(a*v, axis=-2)

def sample_faces(vertices, faces, n_total):
    if len(faces) == 0:
        raise ValueError('Cannot sample points from zero faces.')
    tris = vertices[faces]
    n_faces = len(faces)
    d0 = tris[..., 0:1, :]
    ds = tris[..., 1:, :] - d0
    assert(ds.shape[1:] == (2, 3))
    areas = 0.5 * np.sqrt(np.sum(np.cross(ds[:, 0], ds[:, 1])**2, axis=-1))
    cum_area = np.cumsum(areas)
    cum_area *= (n_total / cum_area[-1])
    cum_area = np.round(cum_area).astype(np.int32)

    positions = []
    last = 0
    for i in range(n_faces):
        n = cum_area[i] - last
        last = cum_area[i]
        if n > 0:
            positions.append(d0[i] + sample_triangle(ds[i], n))
    return np.concatenate(positions, axis=0)

def parse_obj_file(open_file):
    """
    Parse the supplied file.
    Args:
        `open_file`: file-like object with `readlines` method.
    Returns: positions, face_positions, texcoords, face_texcoords, \
        normals, face_normals
    """
    positions = []
    texcoords = []
    normals = []
    face_positions = []
    face_texcoords = []
    face_normals = []

    def parse_face(values):
        if len(values) != 3:
            print('not a triangle', values)
        for v in values:
            for j, index in enumerate(v.split('/')):
                if len(index):
                    if j == 0:
                        face_positions.append(int(index) - 1)
                    elif j == 1:
                        face_texcoords.append(int(index) - 1)
                    elif j == 2:
                        face_normals.append(int(index) - 1)

    parse_fns = {
        'v': lambda values: positions.append([float(x) for x in values]),
        'vt': lambda values: texcoords.append([float(x) for x in values]),
        'vn': lambda values: normals.append([float(x) for x in values]),
        'f': parse_face,
        'mtllib': lambda values: None,
        'o': lambda values: None,
        'usemtl': lambda values: None,
        's': lambda values: None,
        'newmtl': lambda values: None,
        'Ns': lambda values: None,
        'Ni': lambda values: None,
        'Ka': lambda values: None,
        'Kd': lambda values: None,
        'Ks': lambda values: None,
        'd': lambda values: None,
        'illum': lambda values: None,
        'map_Kd': lambda values: None,
    }

    def parse_line(line):
        line = line.strip()
        if len(line) > 0 and line[0] != '#':
            values = line.split(' ')
            code = values[0]
            values = values[1:]
            values = [v for v in values if v is not '']

            if code in parse_fns:
                parse_fns[code](values)

    for lineno, line in enumerate(open_file.readlines()):
        parse_line(line)

    positions = np.array(positions, dtype=np.float32)
    texcoords = np.array(texcoords, dtype=np.float32) if len(texcoords) > 0 \
        else None
    normals = np.array(normals, dtype=np.float32) \
        if len(normals) > 0 else None
    face_positions = np.array(face_positions, dtype=np.uint32).reshape(-1, 3)

    face_texcoords = np.array(face_texcoords, dtype=np.uint32).reshape(-1, 3) \
        if len(face_texcoords) > 0 else None
    face_normals = np.array(face_normals, dtype=np.uint32).reshape(-1, 3) \
        if len(face_normals) > 0 else None

    print('num_vertex=', positions.shape[0], 'num_face=', face_positions.shape[0])

    return positions, face_positions, texcoords, face_texcoords, \
        normals, face_normals

def process_point_cloudv2(path):


    path = os.path.join(path, 'models')
    obj_file = os.path.join(path, 'model_normalized.obj')
    '''
    mesh_list = trimesh.load_mesh(obj_file)
    if not isinstance(mesh_list, list):
        mesh_list = [mesh_list]
    area_sum = 0
    for mesh in mesh_list:
        area_sum += np.sum(mesh.area_faces)

    sample = np.zeros((0, 3), dtype=np.float32)
    normal = np.zeros((0, 3), dtype=np.float32)
    for mesh in mesh_list:
        number = int(round(16384 * np.sum(mesh.area_faces) / area_sum))
        if number < 1:
            continue
        points, index = trimesh.sample.sample_surface_even(mesh, number)
        sample = np.append(sample, points, axis=0)
    '''

    json_file = os.path.join(path, 'model_normalized.json')
    try:
        with open(json_file, 'r') as f:
            meta = json.load(f)
            obj_max = meta['max']
            obj_min = meta['min']
            obj_id = meta['id']

            class_id = obj_file.split('/')[-4]

        out_path = out_name + class_id + '-' + obj_id + '.npy'

        if os.path.exists(out_path):
            print('exist:', out_path)
            return

        with open(obj_file, 'r') as fpc:
            v, f = parse_obj_file(fpc)[0:2]
        sample = sample_faces(v, f, NUM_POINTS)

        print(obj_file, sample.shape, obj_id, class_id)
        np.save(out_path, sample)

    except:
        pass


if __name__ == '__main__':
    paths = glob.glob(shapenet_path + '*/*')
    print('detected obj files=', len(paths))
    num_objs = len(paths)
    pcs = np.zeros((num_objs, NUM_POINTS, 6), dtype=np.float32)
    for i in range(num_objs):
        process_point_cloudv2(paths[i])




