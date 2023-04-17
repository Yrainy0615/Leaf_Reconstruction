import math
import torch

#from kaolin.rep import Mesh
from pytorch3d.io import load_objs_as_meshes , load_obj
from pytorch3d.structures import Meshes

def load_mesh(device,path):
    print("===> Loading Initial Mesh from {}".format(path))
    verts, faces, aux = load_obj(path)
    verts = verts.to(device)
    faces_idx = faces.verts_idx.to(device)
    mesh = Meshes(verts=[verts],faces=[faces_idx])
    return mesh, verts, faces_idx
    

def load_initial_shape(device, path):

    print("===> Loading Initial Mesh from {}".format(path))
    mesh = Meshes.from_obj(path, enable_adjacency=False)
    # mesh.show()
    mesh.to(device)
    vert = mesh.vertices
    vert = vert.unsqueeze(0).to(device)

    return mesh, vert



def rotate_mesh(vert, azi, ele):

    # define rotation matrix
    y_rotation = torch.tensor([[math.cos(math.radians(azi)), 0, math.sin(math.radians(azi))],
                               [0, 1, 0],
                               [-math.sin(math.radians(azi)), 0, math.cos(math.radians(azi))]]).to(vert.device)
    z_rotation = torch.tensor([[math.cos(math.radians(ele)), math.sin(math.radians(ele)), 0],
                               [-math.sin(math.radians(ele)), math.cos(math.radians(ele)), 0],
                               [0, 0, 1]]).to(vert.device)
    rotation_matrix = torch.mm(z_rotation, y_rotation)
    rotation_matrix_t = torch.t(rotation_matrix)
    
    rotate_vert = torch.matmul(vert, rotation_matrix_t.unsqueeze(0))

    return rotate_vert


def locate_mesh(vert, distance, fov, base_size, position, size):
    # distance: distance between camera and plane (leaves are putted on the plane)
    # base_size: size of bush image
    # position: pixel position of leaf on bush image
    # size: size of RoI

    base_x = base_size[0]
    base_y = base_size[1]
    u = position[0]
    v = position[1]
    fov_x = fov[0]
    fov_y = fov[1]

    shift = torch.zeros_like(vert).to(vert.device)
    shift[:, :, 1] += (-2 * distance / base_y) * (v - base_y/2) * math.tan(math.radians(fov_y/2))
    shift[:, :, 2] += (-2 * distance / base_x) * (u - base_x/2) * math.tan(math.radians(fov_x/2))
    located_vert = vert + shift

    return located_vert
