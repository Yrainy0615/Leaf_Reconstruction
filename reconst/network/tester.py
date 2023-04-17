import os
import json
import cv2
import numpy as np
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from kaolin.graphics import DIBRenderer
# from kaolin.rep import TriangleMesh
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.io import save_obj
from network.mesh_net import MeshNet
from data.loader import InstDataset
from util.image import angle_fitting
from util.camera import camera_transform
from util.mesh import rotate_mesh, locate_mesh, load_initial_shape, load_mesh
from util.loss import get_iou


class Tester:
    def __init__(self, args, device):

        self.device = device
        self.init_obj = args.obj
        self.dataset = args.dataset
        self.size = args.size

        self.checkpoint = args.checkpoint
        self.save_dir = args.save_dir

        self.base_x = args.base_x
        self.base_y = args.base_y

        self.data_loader = DataLoader(dataset=InstDataset(self.dataset),
                                      num_workers=args.threads,
                                      batch_size=1,
                                      shuffle=False)

        # Differentiable Renderer
        self.camera_distance = 32
        self.fov_x_base = 40
        self.fov_y_base = 2 * math.degrees(math.atan(math.tan(math.radians(self.fov_x_base / 2)) * self.base_y / self.base_x))
        # self.renderer = DIBRenderer(128, 128, mode='VertexColor').to(self.device)

    def set_network(self, verts_shape):

        network = MeshNet(in_ch=3, num_feat=100, verts_shape=verts_shape).to(self.device)
        network.load_state_dict(torch.load(self.checkpoint))

        return network

    def test(self):

        # init_mesh, init_vert = load_initial_shape(self.device, self.init_obj)
        init_mesh, init_vert , init_face= load_mesh(self.device,self.init_obj )
        network = self.set_network(init_vert.shape)
        mean_def = network.get_mean_deform()

        val_data = dict()

        for iteration, data in enumerate(self.data_loader, 1):
            # load image data and position on the base image
            img, seg, mask = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
            u, v = data[3].item(), data[4].item()

            # shape prediction
            inst_def, texture, scale = network(img)
            textures = TexturesVertex(verts_features=[texture.squeeze(0)])

            pred_vert = scale * (init_vert + mean_def + inst_def)
            pred_mesh = Meshes(verts=[pred_vert.squeeze(0)] ,faces=[init_face],textures=textures)

            min_loss = 1e5
            min_mse = 1e5
            max_iou = 0.0

            # multiplex
            for view in range(6*13):
                azi = (view % 13) * 15  # azimuth
                ele = 90 - (view // 13) * 15  # elevation

                # locate predicted mesh
                rotate_vert = rotate_mesh(pred_vert, azi, ele)
                located_vert = locate_mesh(rotate_vert,
                                           self.camera_distance,
                                           (self.fov_x_base, self.fov_y_base),
                                           (self.base_x, self.base_y),
                                           (u, v),
                                           self.size)

                R, T = look_at_view_transform(dist=self.camera_distance, elev=ele,azim=azi)
                cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
                raster_setting = RasterizationSettings(
                    image_size=128,
                    blur_radius= 0,
                    faces_per_pixel=1,
                )
                
                renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=cameras,
                        raster_settings= raster_setting),
                        shader=SoftPhongShader(
                            device=self.device,cameras=cameras
                        )
                    )
                
                # set camera focused on the mesh
                # camera_params = camera_transform(self.device,
                #                                  self.camera_distance,
                #                                  (self.fov_x_base, self.fov_y_base),
                #                                  (self.base_x, self.base_y),
                #                                  (u, v),
                #                                  self.size)
               #  self.renderer.set_camera_parameters(camera_params)

                # Rendering & angle fitting
                pred_img= renderer(pred_mesh)
                pred_img= pred_img[...,:3]
                pred_mask = 0.299 * pred_img[..., 0] + 0.587 * pred_img[..., 1] + 0.114 * pred_img[..., 2]
                rotate_imgs, rotate_masks, angle = angle_fitting(pred_img.permute(0, 3, 1, 2),
                                                                 pred_mask,
                                                                 mask)

                for i in range(2):
                    rotate_img = rotate_imgs[i]
                    rotate_mask = rotate_masks[i]
                    loss_mse = nn.MSELoss()(rotate_img, seg)
                    iou = get_iou(rotate_mask, mask)
                    loss_iou = 1 - iou
                    loss_img = 10 * loss_mse + 2 * loss_iou

                    if loss_img.item() < min_loss:
                        min_loss = loss_img.item()
                        max_iou = iou.item()
                        min_mse = nn.MSELoss()(rotate_img[rotate_img!=0], seg[rotate_img!=0]).item()
                        min_azimuth = azi
                        min_elevation = ele
                        min_rotation = angle + 180*i

                        # store image and mask for save
                        save_img = rotate_img.permute(0, 2, 3, 1).detach().cpu().numpy()[0] * 255
                        save_mask = rotate_mask.view(self.size, self.size).detach().cpu().numpy() * 255
            save_obj(os.path.join(self.save_dir, 'obj/{:04}.obj'.format(iteration)),verts=pred_vert.squeeze(0),faces=init_face)
            #pred_mesh = TriangleMesh.from_tensors(vertices=pred_vert.squeeze(0), faces=init_mesh.faces)
            #pred_mesh.save_mesh(os.path.join(self.save_dir, 'obj/{:04}.obj'.format(iteration)))
            cv2.imwrite(os.path.join(self.save_dir, 'img/{:04}.png'.format(iteration)), save_img.astype(np.uint8))
            cv2.imwrite(os.path.join(self.save_dir, 'mask/{:04}.png'.format(iteration)), save_mask.astype(np.uint8))
            print("===> Saved {:04}, IoU:{}, MSE:{}, azimuth:{}, elevation:{}, rotation:{}, scale:{}".format(iteration,
                                                                                                             max_iou,
                                                                                                             min_mse,
                                                                                                             min_azimuth,
                                                                                                             min_elevation,
                                                                                                             min_rotation,
                                                                                                             scale.item()))
            val_data["{:04}.obj".format(iteration)] = {'azimuth': min_azimuth,
                                                       'elevation': min_elevation,
                                                       'rotation': min_rotation,
                                                       'scale': scale.item(),
                                                       'iou': max_iou,
                                                       'mse': min_mse}

        with open(os.path.join(self.save_dir, "result.json"), "w") as f:
            json.dump(val_data, f)
