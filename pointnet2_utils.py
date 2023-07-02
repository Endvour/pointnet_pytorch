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

def index_points(points, idx):
    device = points.device

    #get batch_indices
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1]*(len(idx.shape)-1) #view_shape decides the batch_indices.shape[0]
    repeat_shape = list(idx.shape)          #repeat_shape decides the batch_indices.shape[1:]
    repeat_shape[0] = 1                     
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # the same shape of idx
    sampled_points = points[batch_indices, idx, :]

    return sampled_points

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
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # initial sampled index
    distance = torch.ones(B, N).to(device) * 1e10                     # the distance between a point and the sampled point set
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)   # The distance is defined as the minimum distance between the point and the points in the sampled point set
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)       # the xyz of current centroid
        dist = torch.sum((xyz-centroid)**2, dim=-1, dtype=torch.long)  # calculate the distance between all the points and the centroid
        mask = dist < distance                                         # update the distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance,dim=-1)[1]                       # get next centroid index
    
    return centroids