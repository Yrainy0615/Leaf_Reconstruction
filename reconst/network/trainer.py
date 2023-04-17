import os
import rtree
import pandas as pd
import math
import sys
sys.path.append('../data')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import imageio
import cv2
# from kaolin.metrics import trianglemesh
# from kaolin.render.mesh import deftet_sparse_render
# from kaolin.rep import TriangleMesh
# from kaolin.graphics import DIBRenderer
# from kaolin.visualize.vis import show_mesh 
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
from matplotlib import pyplot as plt
import numpy as np
from network.mesh_net import MeshNet
from data.loader import InstDataset
from util.image import angle_fitting
from util.camera import camera_transform
from util.mesh import rotate_mesh, locate_mesh, load_initial_shape,load_mesh
from util.loss import get_iou, get_lap, get_flat, get_def
# from pytorch3d.structures import Meshes

class Trainer:
    def __init__(self, args, device):

        self.device = device
        self.init_obj = args.obj
        self.dataset = args.dataset
        self.size = args.size

        self.lr = args.lr
        self.epochs = args.epochs
        self.checkpoint = args.checkpoint
        self.save_freq = args.save_freq

        self.base_x = args.base_x
        self.base_y = args.base_y

        self.data_loader = DataLoader(dataset=InstDataset(self.dataset),
                                      num_workers=args.threads,
                                      batch_size=1,
                                      shuffle=True)

        self.camera_distance = 32
        self.fov_x_base = 40
        self.fov_y_base = 2 * math.degrees(math.atan(math.tan(math.radians(self.fov_x_base/2)) * self.base_y / self.base_x))
        # raster_settings = RasterizationSettings(
        #         image_size=512, 
        #         blur_radius=0.0, 
        #         faces_per_pixel=1, )
            

# Place a point light in f
        #self.renderer = DIBRenderer(128, 128, mode='VertexColor').to(self.device)
        #self.renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=None, raster_settings=raster_settings  ),  shader=None, )
    def set_network(self, vert_shape, lr):

        network = MeshNet(in_ch=3, num_feat=100, verts_shape=vert_shape).to(self.device)
        optimizer = optim.Adam(network.parameters(), lr=lr)

        return network, optimizer

    def train(self):

        #init_mesh, init_vert = load_initial_shape(self.device, self.init_obj)
        init_mesh, init_vert , init_face= load_mesh(self.device,self.init_obj )
        network, optimizer = self.set_network(init_vert.shape, self.lr)

        # Lists to stack each losses
        mse_stack = []
        iou_stack = []
        flat_stack = []
        lap_stack = []
        inst_stack = []
        mean_stack = []

        for epoch in range(self.epochs):
            total_mse = 0
            total_iou = 0
            total_flat = 0
            total_lap = 0
            total_inst = 0
            total_mean = 0

            for iteration, data in enumerate(self.data_loader, 1):
                # load mean shape, image data and position on the base image
                mean_def = network.get_mean_deform()
                loss_mean_def = get_def(mean_def)
                img, seg, mask = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                u, v = data[3].item(), data[4].item()

                # output shape deformation, texture and scale
                # ==> predict vertices' position of reconstructed mesh
                """
                pred_vert = scale * (init_vert + mean_def + inst_def)
                这行代码的含义是根据输入的参数和已知的均值变形，通过一个线性变换来计算预测的顶点位置。
                 具体来说，init_vert是初始化的顶点坐标，mean_def是整个数据集中所有顶点的均值偏移（即每个顶点在x、y、z三个方向上的平均偏移），inst_def是这个具体样本的顶点偏移。
                在这里，顶点偏移表示顶点在x、y、z三个方向上相对于其初始位置的偏移量。  
                通过将三个偏移量相加，可以得到每个顶点的总偏移量，然后通过scale将其缩放到正确的大小。
                #这个scale是在模型训练过程中学习到的，用于将3D模型缩放到正确的大小。
                """
                inst_def, texture, scale = network(img)
                textures = TexturesVertex(verts_features=[texture.squeeze(0)])
                pred_vert = scale * (init_vert + mean_def + inst_def)
                pred_mesh = Meshes(verts=[pred_vert.squeeze(0)] ,faces=[init_face],textures=textures)
                min_loss = 1e5
                min_mse = 1e5
                min_iou = 1.0

                # pose(view)-multiplex
                # define range depend on number of candidates
                
                loss_stack = torch.zeros(6 * 13).to(self.device)
                for view in range(6*13):   #分割视角
                    azi = (view % 13) + 15  # azimuth
                    ele = 90 - (view // 13) * 15  # elevation

                    # locate predicted mesh
                    # 这个函数实现的功能是将一个mesh对象的顶点坐标绕y轴和z轴旋转一定角度。其中，azi是绕y轴旋转的角度，ele是绕z轴旋转的角度。
                    rotate_vert = rotate_mesh(pred_vert, azi, ele)
                    
                    """
                    这段代码的作用是将三维模型的顶点位置根据相机的位置和视角调整到图像平面上的正确位置，以便进行渲染。
                    具体来说，这个函数通过计算相机和物体之间的距离和相机视角的大小，
                    计算出物体相对于相机的位置，然后将顶点位置根据这个位置调整。
                    """    
                    located_vert = locate_mesh(rotate_vert,
                                               self.camera_distance,
                                               (self.fov_x_base, self.fov_y_base),
                                               (self.base_x, self.base_y),
                                               (u, v),
                                               self.size)

                    # set camera focused on the mesh
                    # camera_params = [camera_view_mtx, camera_view_shift, camera_projection_mtx]
                    R, T = look_at_view_transform(dist=self.camera_distance, elev=ele,azim=azi)
                    # camera_params = camera_transform(self.device,
                    #                                  self.camera_distance,
                    #                                  (self.fov_x_base, self.fov_y_base),
                    #                                  (self.base_x, self.base_y),
                    #                                  (u, v),
                    #                                  self.size)
                    # self.renderer.set_camera_parameters(camera_params)
                       
                            
                    cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
                    lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
                    raster_settings = RasterizationSettings(
                        image_size=128, 
                        blur_radius=0.0, 
                        faces_per_pixel=1, 
                    )
                    renderer = MeshRenderer(
                        rasterizer=MeshRasterizer(
                        cameras=cameras, 
                         raster_settings=raster_settings),
                        shader=SoftPhongShader(
                            device=self.device,cameras=cameras
                        ))       
                    #rastrizer = MeshRasterizer(cameras=cameras,raster_settings=raster_settings)
                   # fragments = rastrizer(pred_mesh)
                    pred_img = renderer(pred_mesh,)
                    pred_img = pred_img[...,:3]
                    pred_mask = 0.299 * pred_img[..., 0] + 0.587 * pred_img[..., 1] + 0.114 * pred_img[..., 2]  # shape: (1, 128, 128)
                    # pred_img: (B, H, W, C)
                    # pred_mask: (B, H, W, 1)           
                    # pred_img, pred_mask, pred_norms = self.renderer(points=[located_vert, init_mesh.faces.long()], colors_bxpx3=colors)
                   
                    # rotate target image to fit angle
                    # output 2 rotated images because of 180 degree ambiguity
             

                    #cv2.imshow(pred_mask.squeeze(3),'mask')
                    target_imgs, target_masks, angle = angle_fitting(seg,
                                                                     mask,
                                                                     pred_mask)
               
               
                    
                    rotation_stack = torch.zeros(2)
                    for i in range(2):
                        target_img = target_imgs[i].permute(0, 2, 3, 1)
                        target_mask = target_masks[i]
                        loss_mse = nn.MSELoss()(target_img, pred_img)
                        iou = get_iou(target_mask.unsqueeze(3), pred_mask)
                        loss_iou = 1 - iou
                        loss_img = 100 * loss_mse + 2 * loss_iou
                        rotation_stack[i] = loss_img

                        if loss_img.item() < min_loss:
                            min_loss = loss_img.item()
                            min_mse = loss_mse.item()
                            min_iou = loss_iou.item()

                    # use minimum image loss for optimization
                    loss_stack[view] = torch.min(rotation_stack)
              
                # other loss
                # loss_flat = get_flat(init_mesh, pred_norms.squeeze(0))
                loss_lap = get_lap(pred_mesh, init_mesh)
                loss_inst_def = get_def(inst_def.squeeze(0))
                loss = torch.min(loss_stack) + 0.5 * loss_lap + 10 * loss_inst_def #+ 0.5 * loss_flat 

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_mse += min_mse
                total_iou += min_iou
                # total_flat += loss_flat.item()
                total_lap += loss_lap.item()
                total_mean += loss_mean_def.item()
                total_inst += loss_inst_def.item()
            print('Epoch[{}/{}]: MSE:{}, IoU:{}, Norm:{}, Lap:{}, InstDef:{}, MeanDef:{}'.format(epoch + 1,
                                                                                                 self.epochs,
                                                                                                 total_mse / len(self.data_loader),
                                                                                                 total_iou / len(self.data_loader),
                                                                                                 total_flat / len(self.data_loader),
                                                                                                 total_lap / len(self.data_loader),
                                                                                                 total_inst / len(self.data_loader),
                                                                                                 total_mean / len(self.data_loader)))

            mse_stack.append(total_mse / len(self.data_loader))
            iou_stack.append(total_iou / len(self.data_loader))
            flat_stack.append(total_flat / len(self.data_loader))
            lap_stack.append(total_lap / len(self.data_loader))
            mean_stack.append(total_mean / len(self.data_loader))
            inst_stack.append(total_inst / len(self.data_loader))

            if (epoch+1) % self.save_freq == 0:
                # save mean shape
                mean_shape_deform = network.get_mean_deform()
                mean_vert = init_vert + mean_shape_deform
                # mean_mesh = Meshes(verts=[mean_vert.squeeze(0)], faces=[init_face])
                os.makedirs(os.path.join(self.checkpoint, 'mean'), exist_ok=True)
                save_obj(f=os.path.join(self.checkpoint, 'mean/{:04}.obj'.format(epoch+1)), verts=mean_vert.squeeze(0),faces=init_face)

                # save parameters
                model_path = os.path.join(self.checkpoint, 'epoch{}.pth'.format(epoch+1))
                torch.save(network.state_dict(), model_path)

                # save log
                df_train = pd.DataFrame({'mse': mse_stack,
                                         'iou': iou_stack,
                                         'flat': flat_stack,
                                         'lap': lap_stack,
                                         'mean_def': mean_stack,
                                         'inst_def': inst_stack})
                log_path = os.path.join(self.checkpoint, 'log.csv')
                df_train.to_csv(log_path)
                print("===> Saved Log in {}".format(self.checkpoint))
