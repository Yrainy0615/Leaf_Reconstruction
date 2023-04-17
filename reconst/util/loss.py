import torch
from pytorch3d.loss import  mesh_edge_loss, mesh_laplacian_smoothing

def get_iou(pred: torch.Tensor, target: torch.Tensor):

    intersec = torch.sum(pred * target)
    union = torch.sum((pred + target) - pred * target)
    iou = intersec / union

    return iou


def get_flat(mesh, norms):

    loss = 0
    for i in range(3):
        norm1 = norms
        norm2 = norms[mesh.ff[:, i]]
        cos = torch.sum(norm1 * norm2, dim=1)
        loss += torch.mean((cos - 1) ** 2)
    loss /= 2
    return loss


def get_def(delta_v):  # delta_v = (1, V, 3)

    loss = torch.mean(torch.norm(delta_v, p=2, dim=1))

    return loss


def get_lap(mesh1, mesh2):
    # Returns the change in laplacian over two meshes
    # Args:
    #     mesh1 (Mesh): first mesh
    #     mesh2: (Mesh): second mesh
    # Returns:
    #     lap_loss (torch.Tensor):  laplacian change over the mesh
    # Example:
    #     >>> mesh1 = TriangleMesh.from_obj(file)
    #     >>> mesh2 = TriangleMesh.from_obj(file)
    #     >>> mesh2.vertices = mesh2.vertices * 1.05
    #     >>> lap = laplacian_loss(mesh1, mesh2)
    #
    lap1 = mesh_laplacian_smoothing(mesh1)
    # lap1 = mesh1.compute_laplacian()
    # lap2 = mesh2.compute_laplacian()
    
    lap2 = mesh_laplacian_smoothing(mesh2)
    lap_loss = torch.sum((lap1 - lap2)**2) / (lap1.numel())
    #lap_loss = torch.mean(torch.sum((lap1.item() - lap2.item())**2, 1))
    return lap_loss

