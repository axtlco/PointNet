import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.
        x = pointcloud.transpose(2, 1)
        input_trans = None
        feat_trans = None

        if self.input_transform:
            input_trans = self.stn3(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, input_trans)
            x = x.transpose(2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.feature_transform:
            feat_trans = self.stn64(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, feat_trans)
            x = x.transpose(2, 1)

        point_feat = x
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        global_feat = torch.max(x, 2)[0]

        return global_feat, point_feat, input_trans, feat_trans


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512))
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256))
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        global_feat, _, input_trans, feat_trans = self.pointnet_feat(pointcloud)
        x = F.relu(self.fc1(global_feat))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)
        return logits, input_trans, feat_trans


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.pointnet_feat = PointNetFeat(input_transform=True, feature_transform=True)
        self.conv1 = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512))
        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128))
        self.conv5 = nn.Conv1d(128, m, 1)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        global_feat, point_feat, input_trans, feat_trans = self.pointnet_feat(pointcloud)
        num_points = pointcloud.shape[1]
        global_feat = global_feat.unsqueeze(-1).repeat(1, 1, num_points)
        x = torch.cat([point_feat, global_feat], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        logits = self.conv5(x)
        return logits, input_trans, feat_trans



def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
