import deep_fixel
from datetime import datetime

lr = 1e-3
batch_size = 512
loss = "MSE"
model = "mesh_mlp"
gpu_id = 0
seed = 42
mesh_subdivide = 1
healpix = True
kappa = 100
min_separation_angle = 0
n_fibers = 'both'
model_path = "./models/pretrained/best_model_mlp.pth"
test_dir = "./test_data"
output_dir = './outputs/pretrained_mlp'
amp_threshold = 0.1

deep_fixel.test_mesh_model(
    model=model,
    model_path=model_path,
    batch_size=batch_size,
    n_fibers=n_fibers,
    subdivide_mesh=mesh_subdivide,
    amp_threshold=amp_threshold,
    output_dir=output_dir,
    kappa=kappa,
    test_dir=test_dir,
    gpu_id=gpu_id,
    healpix=healpix,
    use_dipy=True,
    min_separation_angle=min_separation_angle,
    is_symmetric=True
)

lr = 1e-3
batch_size = 512
loss = "MSE"
model = "mesh_scnn"
gpu_id = 0
seed = 42
mesh_subdivide = 1
healpix = True
kappa = 100
min_separation_angle = 0
n_fibers = 'both'
model_path = "./models/pretrained/best_model_scnn.pth"
test_dir = "./test_data"
output_dir = './outputs/pretrained_scnn'
amp_threshold = 0.1

deep_fixel.test_mesh_model(
    model=model,
    model_path=model_path,
    batch_size=batch_size,
    n_fibers=n_fibers,
    subdivide_mesh=mesh_subdivide,
    amp_threshold=amp_threshold,
    output_dir=output_dir,
    kappa=kappa,
    test_dir=test_dir,
    gpu_id=gpu_id,
    healpix=healpix,
    use_dipy=True,
    min_separation_angle=min_separation_angle,
    is_symmetric=True
)
