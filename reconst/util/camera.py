import numpy as np
import math
import torch
# from kaolin.graphics.dib_renderer.utils import perspectiveprojectionnp

def compute_camera_params(device, targetW, fov_y):

    camera_projection_mtx = perspectiveprojectionnp(fov_y, 1.0)
    camera_projection_mtx = torch.FloatTensor(camera_projection_mtx).to(device)

    camera_view_mtx = []
    camera_view_shift = []
    # position
    camX = 32
    camY = 0
    camZ = 0
    cam_pos = np.array([camX, camY, camZ])

    # angle
    axisZ = cam_pos.copy() - targetW
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    l2[l2 == 0] = 1
    cam_mat = cam_mat / np.expand_dims(l2, 1)

    camera_view_mtx.append(torch.FloatTensor(cam_mat))
    camera_view_shift.append(torch.FloatTensor(cam_pos))
    camera_view_mtx = torch.stack(camera_view_mtx).to(device)
    camera_view_shift = torch.stack(camera_view_shift).to(device)

    camera_params = [camera_view_mtx, camera_view_shift, camera_projection_mtx]

    return camera_params


def camera_transform(device, distance, fov, base_size, position, size):

    fov_x = fov[0]
    fov_y = fov[1]
    base_x = base_size[0]
    base_y = base_size[1]
    u = position[0]
    v = position[1]

    targetW = np.array([0,
                        (-2*distance / base_y) * (v - base_y/2) * math.tan(math.radians(fov_y/2)),
                        (-2*distance / base_x) * (u - base_x/2) * math.tan(math.radians(fov_x/2))])
    fov_ratio = size / base_y
    new_fov = 2 * math.degrees(math.atan(math.tan(math.radians(fov_y/2)) * fov_ratio))
    camera_params = compute_camera_params(device, targetW, new_fov * np.pi / 180)

    return camera_params
