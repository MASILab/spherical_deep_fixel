import deep_fixel
from datetime import datetime

lr = 1e-3
batch_size = 512
n_steps = 20000
validation_patience = 5
loss = "MSE"
model = "mesh_scnn"
gpu_id = 0
seed = 42
mesh_subdivide = 1
kappa = 100
n_fibers = 'both'
healpix = True
csd = True
snr = 30
save_dir = "./models"
test_dir = "./test_data"

datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = f"deepfixel_{model}_noise_{datetime_str}"

deep_fixel.train_mesh_model(
    run_name=name,
    lr=lr,
    batch_size=batch_size,
    n_steps=n_steps,
    validation_patience=validation_patience,
    loss_name=loss,
    model=model,
    gpu_id=gpu_id,
    seed=seed,
    mesh_subdivide=mesh_subdivide,
    kappa=kappa,
    n_fibers=n_fibers,
    save_dir=save_dir,
    healpix=healpix,
    csd=csd,
    snr=snr,
)

output_dir = f'./outputs/{name}'

amp_threshold = 0.1
model_path = f"./models/{name}/best_model.pth"

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
)
