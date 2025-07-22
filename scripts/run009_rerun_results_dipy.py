import deep_fixel
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import time 

lr = 1e-3
batch_size = 512
loss = "MSE"
model = "mesh_scnn"
gpu_id = 0
seed = 42
mesh_subdivide = 1
kappa = 100
n_fibers = 'both'
healpix = True
amp_threshold = 0.0
min_separation_angle = 0.0
save_dir = "../models"
test_dir = "../test_data"

name = "deepfixel_mesh_scnn_healpix_2025-04-14_12-21-03"

output_dir = f'../outputs/{name}'
model_path = f"../models/{name}/best_model.pth"

start_time = time.time()

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
model = "mesh_mlp"
gpu_id = 0
seed = 42
mesh_subdivide = 1
kappa = 100
n_fibers = 'both'
healpix = True
amp_threshold = 0.0
min_separation_angle = 0.0
save_dir = "../models"
test_dir = "../test_data"

name = "deepfixel_mesh_mlp_healpix_2025-04-15_08-32-03"

output_dir = f'../outputs/{name}'
model_path = f"../models/{name}/best_model.pth"

start_time = time.time()

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
