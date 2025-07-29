from torch import nn
from hsd.model.unet import GraphCNNUnet
from hsd.utils.sampling import HealpixSampling

class CrossingFiberMeshMLP(nn.Module):
    def __init__(self, n_mesh, device='cuda', sphere=None):
        """Crossing fiber MLP

        Parameters
        ----------
        n_mesh : int
            Number of points in mesh
        norm : bool, optional
            Whether to norm output to integrate to 1, by default False
        device : str, optional
            If norm, must define device, by default 'cuda'
        sphere : Dipy sphere, optional
            If norm, must define sphere that is the VERTICES of the icosphere (not face sphere!) to norm on, by default None

        Raises
        ------
        ValueError
            _description_
        """
        super(CrossingFiberMeshMLP, self).__init__()
        self.n_mesh = n_mesh
        
        # Create a basic MLP with 5 layers: 4 hidden layers and 1 output layer with n_fibers * n_coeff neurons
        self.mlp = nn.Sequential(
            nn.Linear(self.n_mesh, 250),
            nn.ReLU(),
            nn.Linear(250, 150),
            nn.ReLU(),
            nn.Linear(150, 50),
            nn.ReLU(),
            nn.Linear(50, 150),
            nn.ReLU(),
            nn.Linear(150, 250),
            nn.ReLU(),
            nn.Linear(250, self.n_mesh),
        )

    def forward(self, x):
        out = self.mlp(x)

        # Reshape to n_fibers x n_coeff
        out = out.view(-1, self.n_mesh)    

        return out
    
class CrossingFiberMeshSCNN(nn.Module):
    def __init__(self, device='cuda', n_side=8, depth=5, patch_size=1, sh_degree=6, pooling_mode='average', pooling_name='spherical', use_hemisphere=True,
                 in_channels=1, out_channels=1, filter_start=2, block_depth=1, in_depth=1, kernel_sizeSph=3, kernel_sizeSpa=3, isoSpa=True, keepSphericalDim = True):
        """Crossing fiber spherical CNN based on github.com/AxelElaldi/fast-equivariant-deconv

        Parameters
        ----------
        device : str, optional
            PyTorch device, by default 'cuda'
        n_side : int, optional
            Number of sides for Healpix sampling, by default 8
        depth : int, optional
            Number of hierarchical levels for sampling, by default 5
        patch_size : int, optional
            Spatial patch size, by default 1
        sh_degree : int, optional
            Spherical harmonics maximum order, by default 6
        pooling_mode : str, optional
            'average' or 'max, by default 'average'
        pooling_name : str, optional
            'spatial', 'spherical', 'mixed', or 'bekkers', by default 'spherical'
        use_hemisphere : bool, optional
            Whether to only define on hemisphere, by default True
        in_channels : int, optional
            Number of convolutional channels on input, by default 1
        out_channels : int, optional
            Number of convolutional channels on output, by default 1
        filter_start : int, optional
            Number of filters after first convolution (doubles after each pooling), by default 2
        block_depth : int, optional
            Number of blocks between two poolins for encoder, by default 1
        in_depth : int, optional
            Number of blocks before unpooling for decoder, by default 1
        kernel_sizeSph : int, optional
            Spherical kernel size, by default 3
        kernel_sizeSpa : int, optional
            Spatial kernel size, by default 3
        isoSpa : bool, optional
            Use isotropic spatial filter for E3 equivariance, by default True
        keepSphericalDim : bool, optional
            Keep spherical dimension (or average over output vertices if False), by default True
        """
        super(CrossingFiberMeshSCNN, self).__init__()
        self.sampling = HealpixSampling(n_side, depth, patch_size, sh_degree, pooling_mode, pooling_name, use_hemisphere)
        laps = self.sampling.laps
        poolings = self.sampling.pooling
        vec = self.sampling.vec
        patch_size_list = self.sampling.patch_size_list
        nvec_out = self.sampling.sampling.vectors.shape[0]
        self.unet = GraphCNNUnet(in_channels, out_channels, filter_start, block_depth, in_depth, kernel_sizeSph, kernel_sizeSpa, poolings, laps, pooling_name, isoSpa, keepSphericalDim, patch_size_list, vec, nvec_out)

    def forward(self, x):
        # Reshape to B x 1 x n_mesh x 1 x 1 x 1 (no spatial dims)
        x = x.view(x.shape[0], 1, x.shape[1], 1, 1, 1) 
        out = self.unet(x)

        # Reshape to B x n_mesh
        out = out.view(out.shape[0], out.shape[2])
        return out
