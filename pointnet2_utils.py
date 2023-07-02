import torch

def square_distance(src, dst):
    """
    calculate the square dsitance between each two points
    dist = (xn-xm)**2 + (yn-ym)**2 + (zn-zm)**2
         = (xn**2 + yn**2 + zn**2) + (xm**2+ym**2+zm**2) -2*(xm*xn + ym*yn + zm*zn)
    (xn**2 + yn**2 + zn**2) = torch.sum(src**2, dim=-1)
    (xm**2 + ym**2 + zm**2) = torch.sum(dist**2, dim=-1)
    (xm*xn + ym*yn + zm*zn) = -2 * (src*dist^T)

    Input:
        src(Tensor[B, N, C]): source points
        dst(Tensor[B, M, C]): target points
    Output:
        dist(Tensor[B, N, M]): square distance between each two points
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)

    return dist

def farthest_point_sample(xyz, npoint):
    '''
    Input:
        xyz(Tensor[B, N, 3]): pointcloud data
        npoint(int): number of samples
    
    Output:
        centroids(Tensor[B, npoint]): sampled pointcloud index
    '''
    device = xyz.device
    B, N, _ = xyz.shape
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = centroids[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz-centroid)**2, dim=-1, dtype=torch.long)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance,dim=-1)[1]
    
    return centroids