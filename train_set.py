import os
# Workaround for Windows: avoid crash when multiple OpenMP runtimes are loaded
# (e.g., numpy/torch/other libs pulling in different libiomp5md.dll copies).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(" -- GPU training -- ")


def shrink_mask(mask, kernel_size=3,kernel_size2=7):
    """通过边界去除实现缩小"""
    # 无滑移边界
    dilation = F.max_pool2d(1-mask, 3, stride=1, padding=kernel_size // 2)
    boundary = mask-1+dilation
    # 计算物理方程的内域
    dilation2 = F.max_pool2d(1-mask, 7, stride=1, padding=kernel_size2 // 2)
    boundary2 = mask-1+dilation2
    inner_mask=mask- boundary2

    return boundary,inner_mask


class ModeLoss(nn.Module):
    def __init__(self, lambda_data=10, lambda_phy=0.01, lambda_bnd=1.0, lambda_ext=10.0):
        super().__init__()
        # 最大权重（预热终值），可通过构造函数调整
        self.lambda_phy_max = lambda_phy
        self.lambda_bnd_max = lambda_bnd
        self.lambda_ext_max = lambda_ext
        self.lambda_data = lambda_data

        # 当前 epoch 实际使用的权重（运行时更新）
        self.lambda_phy = 0.0
        self.lambda_bnd = 0.0
        self.lambda_ext = 0.0

        # 二阶导数卷积核（归一化坐标）

        self.lap_kernel = torch.tensor([
            [[[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]]]
        ])/((1/255) ** 2)  # 坐标归一化补偿

    def forward (self, pred, phy_pred, mask, target, epoch,epochs, mode):
        """计算损失

        - ML / SGML 模式: 仅使用 SmoothL1 数据损失
        - PINN 模式: 数据损失 + 边界损失 + 外部区域损失 + 物理方程损失
        """
        U_ref = 1.6384 # 参考速度
        U_norm = 0.03451   # 归一化速度(训练集最大值)
        u_nd = phy_pred * mask * U_norm / U_ref  # 归一化速度场
        smooth_l1 = nn.SmoothL1Loss()

        data_loss = smooth_l1(pred, target)
        # 纯数据驱动模式: 只返回数据损失
        if mode in ["ML", "SGML"]:
            return data_loss, {"data_loss": data_loss.item()}

        # PINN 模式: 物理约束 + 数据
        # 1. 边界损失（无滑移条件）
        boundary_mask, inner_mask = shrink_mask(mask)
        boundary_points = pred * boundary_mask
        boundary_loss = torch.sqrt(torch.sum(boundary_points ** 2) / torch.sum(boundary_mask))
        
        # 2. 外部区域损失（强制为0）
        exterior_loss = torch.sqrt(torch.sum((pred * (1 - mask)) ** 2) / (torch.sum(1 - mask)))

        # 3. 物理方程损失
        laplacian = F.conv2d(u_nd,self.lap_kernel.to(u_nd.device), padding=1,)
        residual = (laplacian + 1) * inner_mask  # μ∇²w + dp/dz = 0
        physics_loss = smooth_l1(residual, torch.zeros_like(residual))

        # 权重随 epoch 平滑预热: 200 之前为 0，200~400 线性升至最大值，400 之后保持
        # phase ∈ [0,1]: epoch<=200→0，200<epoch<400→线性插值，epoch>=400→1
        phase = min(max(epoch - epochs/10, 0), epochs/10) / (epochs/10)

        self.lambda_phy = self.lambda_phy_max * phase
        self.lambda_bnd = self.lambda_bnd_max * phase
        self.lambda_ext = self.lambda_ext_max * phase

        total_loss = (
            self.lambda_data * data_loss
            + self.lambda_bnd * boundary_loss
            + self.lambda_ext * exterior_loss
            + self.lambda_phy * physics_loss
        )

        return total_loss, {
            "data_loss": data_loss.item(),
            "boundary_loss": boundary_loss.item(),
            "exterior_loss": exterior_loss.item(),
            "physics_loss": physics_loss.item(),
        }


class MyDataset(Dataset):
    def __init__(self, root, subfolder, transform=None):
        """
        自定义数据集初始化
        :param root: 数据文件根目录
        :param subfolder: 合并后的 .npy 文件名/相对路径（形状: N*256*768）
        """
        super(MyDataset, self).__init__()
        self.path = os.path.normpath(os.path.join(root, subfolder))
        self.transform = transform

        self._mmap = None
        arr = np.load(self.path, mmap_mode='r')
        self._length = arr.shape[0]
        del arr
  
    def __len__(self):
        """
        :return: 数据集大小
        """

        return self._length

    def __getitem__(self, item):
        """
        支持索引以便dataset可迭代获取
        :param item: 索引
        :return: 索引对应的数据单元
        """
        if self._mmap is None:
            self._mmap = np.load(self.path, mmap_mode='r')

        train_data = np.array(self._mmap[item], copy=True)

        if self.transform is not None:
            train_data=self.transform(train_data)

        return train_data, 
## 加载数据##
def loadData(root, subfolder, batch_size, shuffle=True):
    """
    加载数据以返回DataLoader类型
    :param root: 数据文件根目录
    :param subfolder: 数据文件子目录
    :param batch_size: 批处理样本大小
    :param shuffle: 是否打乱数据（默认为是）
    :return: DataLoader类型的可迭代数据
    """
    # 数据预处理方式
    transform = transforms.Compose([transforms.ToTensor() ])# 创建Dataset对象
    dataset = MyDataset(root, subfolder,  transform=transform)
    print('data_num:',len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


## 训练生成器 ##
def G_train(G, X, LOSS, optimizer_G, epoch, epochs, mode):
    """训练模型

    数据约定: X 的尺寸为 [B, C, 256, 768]，其中按宽度切分为三段:
    - 第一个 256*256:  approximate
    - 第二个 256*256: mask
    - 第三个 256*256: target

    不同模式下输入 G 的 x 定义为:
    - SGML 模式: x = 近似场
    - ML / PINN 模式: x = mask
    """

    X = X.to(torch.float)

    # 按宽度方向切三块: [approx | mask | target]
    approx = X[:, :, :, 0:256].to(device)
    mask = X[:, :, :, 256:512].to(device)
    y = X[:, :, :, 512:768].to(device)

    mode = str(mode).upper()
    if mode == "SGML":
        x = approx
    else:  # ML 和 PINN
        x = mask

    # 梯度初始化为0
    G.zero_grad()

    G_output = G(x)
    G.eval()
    pred_phy = G(x)  # 用于 physics_loss（无 Dropout，更平滑）
    G.train()

    G_loss, loss_dict = LOSS(G_output, pred_phy, mask, y, epoch,epochs, mode)
    # 反向传播并优化

    G_loss.backward()
    optimizer_G.step()

    return G_loss.data.item(),loss_dict



