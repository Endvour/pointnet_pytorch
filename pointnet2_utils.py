import torch
import torch.nn as nn
import torch.nn.functional as F

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

def query_ball_points(radius, nsample, xyz, new_xyz):
    '''
    
    '''
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    device = xyz.device
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1,1,N).repeat(B,S,1)
    distance = square_distance(new_xyz, xyz)
    group_idx[distance > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, nsample]
    group_first = group_idx[:,:,0].view(B, S, 1).repeat(1,1,nsample)
    mask = group_idx==N
    group_idx[mask] = group_first[mask]
    
    return group_idx

def sample_and_gruop(npoint, radius, nsample, xyz, points, returnfps=False):
    '''
    
    '''
    B, _, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_points(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) #[B, S, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
        
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    
    device =xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
        
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, npoint]
            new_points_concat: sample points feature data, [B, D', npoint]
        """
        xyz = xyz.permute(0, 2, 1) # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1) # [B, N, D]
        
        if self.group_all:
            # new_xyz [B, 1, C]
            # new_points [B, 1, N, C+D]
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # new_xyz: [B, npoint, C]
            # new_points: [B, npoint, nsample, C+D]
            new_xyz, new_points = sample_and_gruop(self.npoint, self.radius, self.nsample, xyz, points)
            
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # max pooling
        new_points = torch.max(new_points, dim=2)[0] #[B, D', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)# [B, C, npoint]
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            mlp_convs = nn.ModuleList()
            mlp_bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            
            self.conv_blocks.append(mlp_convs)
            self.bn_blocks.append(mlp_bns)
    
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # [B, N, C]
        B, N, C = xyz.shape
        S = self.npoint
        if points is not None:
            points = points.permute(0, 2, 1) # [B, N, D]
        
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint)) # [B, S, C]
        new_point_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_points(radius, K, xyz, new_xyz) # [B, S, K]
            grouped_xyz = index_points(xyz, group_idx) # [B, S, K, C]
            grouped_xyz -= new_xyz.view(B, S, 1, C)    # [B, S, K, C]
            
            if points is not None:
                grouped_points = index_points(points, group_idx) # [B, S, K, D]
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1) # [B, S, K, C+D]
            else:
                grouped_points = grouped_xyz # [B, S, K, C]
                
            grouped_points = grouped_points.permute(0, 3, 2, 1) # [B, C+D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, dim=2)[0] #[B, D', S]
            new_point_list.append(new_points)
        
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_point_list, dim=1)
        
        return new_xyz, new_points_concat
            
        