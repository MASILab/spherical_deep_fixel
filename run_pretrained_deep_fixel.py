import deep_fixel
from datetime import datetime

lr = 1e-3
batch_size = 512
n_steps = 10000
validation_patience = 5
loss = "MSE"
model = "mesh_mlp"
gpu_id = 1
seed = 42
mesh_subdivide = 3
kappa = 100
n_fibers = 'both'
model_path = "./models/pretrained/best_model.pth"
test_dir = "./test_data"
output_dir = './outputs/pretrained'
amp_threshold = 0.1

deep_fixel.test_mesh_model(
    model_path=model_path,
    batch_size=batch_size,
    n_fibers=n_fibers,
    subdivide_mesh=mesh_subdivide,
    amp_threshold=amp_threshold,
    output_dir=output_dir,
    kappa=kappa,
    test_dir=test_dir,
    gpu_id=gpu_id,
)
