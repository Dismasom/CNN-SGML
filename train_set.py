import os
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
    # No-slip boundary
    dilation = F.max_pool2d(1-mask, 3, stride=1, padding=kernel_size // 2)
    boundary = mask-1+dilation
    # Calculate inner domain for physical equations
    dilation2 = F.max_pool2d(1-mask, 7, stride=1, padding=kernel_size2 // 2)
    boundary2 = mask-1+dilation2
    inner_mask=mask- boundary2

    return boundary,inner_mask


class ModeLoss(nn.Module):
    def __init__(self, lambda_data=10, lambda_phy=0.01, lambda_bnd=1.0, lambda_ext=10.0):
        super().__init__()
        # Maximum weights (final values after warmup)
        self.lambda_phy_max = lambda_phy
        self.lambda_bnd_max = lambda_bnd
        self.lambda_ext_max = lambda_ext
        self.lambda_data = lambda_data

        # Current epoch actual used weights
        self.lambda_phy = 0.0
        self.lambda_bnd = 0.0
        self.lambda_ext = 0.0

        # Second-order derivative convolution kernel (normalized coordinates)

        self.lap_kernel = torch.tensor([
            [[[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]]]
        ])/((1/255) ** 2)  # Coordinate normalization compensation

    def forward (self, pred, phy_pred, mask, target, epoch,epochs, mode):
        """Calculate loss

        - ML / SGML / FNO mode: Only use data loss
        - PINN mode: Data loss + Boundary loss + Exterior region loss + Physics equation loss
        """
        U_ref = 1.6384 # Reference velocity (for non-dimensionalization)
        U_norm = 0.03451   # Normalized velocity (maximum value in training set)
        u_nd = phy_pred * mask * U_norm / U_ref  # Normalized velocity field
        smooth_l1 = nn.SmoothL1Loss()

        data_loss = smooth_l1(pred, target)
        # Only return data loss
        if mode in ["ML", "SGML"]:
            return data_loss, {"data_loss": data_loss.item()}

        # PINN mode: Physics constraints + Data
        # 1. Boundary loss (no-slip condition)
        boundary_mask, inner_mask = shrink_mask(mask)
        boundary_points = pred * boundary_mask
        boundary_loss = torch.sqrt(torch.sum(boundary_points ** 2) / torch.sum(boundary_mask))
        
        # 2. Exterior region loss (forced to 0)
        exterior_loss = torch.sqrt(torch.sum((pred * (1 - mask)) ** 2) / (torch.sum(1 - mask)))

        # 3. Physics equation loss
        laplacian = F.conv2d(u_nd,self.lap_kernel.to(u_nd.device), padding=1,)
        residual = (laplacian + 1) * inner_mask  # μ∇²w + dp/dz = 0
        physics_loss = smooth_l1(residual, torch.zeros_like(residual))

        # Weight smooth warmup with epoch
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
        Custom dataset initialization
        :param root: Data file root directory
        :param subfolder: Merged .npy file name/relative path (shape: N*256*768)
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
        :return: Dataset size
        """

        return self._length

    def __getitem__(self, item):
        """
        Support indexing so dataset can be iterated
        :param item: Index
        :return: Data unit corresponding to index
        """
        if self._mmap is None:
            self._mmap = np.load(self.path, mmap_mode='r')

        train_data = np.array(self._mmap[item], copy=True)

        if self.transform is not None:
            train_data=self.transform(train_data)

        return train_data, 
## Load data ##
def loadData(root, subfolder, batch_size, shuffle=True):
    """
    Load data to return DataLoader type
    :param root: Data file root directory
    :param subfolder: Data file subdirectory
    :param batch_size: Batch sample size
    :param shuffle: Whether to shuffle data (default is True)
    :return: Iterable data of DataLoader type
    """
    # Data preprocessing method
    transform = transforms.Compose([transforms.ToTensor() ])# Create Dataset object
    dataset = MyDataset(root, subfolder,  transform=transform)
    print('data_num:',len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


## Train Generator ##
def G_train(G, X, LOSS, optimizer_G, epoch, epochs, mode):
    """Train model

    Data convention: X has size [B, C, 256, 768], divided into three parts by width:
    - First 256*256:  approximate
    - Second 256*256: mask
    - Third 256*256: target

    Input x to G in different modes is defined as:
    - SGML mode: x = approximate field
    - ML / PINN / FNO mode: x = mask
    """

    X = X.to(torch.float)

    # Cut into three parts by width: [approx | mask | target]
    approx = X[:, :, :, 0:256].to(device)
    mask = X[:, :, :, 256:512].to(device)
    y = X[:, :, :, 512:768].to(device)

    mode = str(mode).upper()
    if mode == "SGML":
        x = approx
    else:
        x = mask

    # Initialize gradient to 0
    G.zero_grad()

    G_output = G(x)
    G.eval()
    pred_phy = G(x)  # For physics_loss (no Dropout, smoother)
    G.train()

    G_loss, loss_dict = LOSS(G_output, pred_phy, mask, y, epoch,epochs, mode)
    # Backpropagation and optimization

    G_loss.backward()
    optimizer_G.step()

    return G_loss.data.item(),loss_dict



