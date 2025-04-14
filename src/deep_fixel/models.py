from torch import nn

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