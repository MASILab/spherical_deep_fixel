import deep_fixel
from datetime import datetime
from argparse import ArgumentParser
import yaml

parser = ArgumentParser(description="Train DeepFixel model")
parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

lr = float(config.get('lr', 1e-3))
batch_size = int(config.get('batch_size', 512))
n_steps = int(config.get('n_steps', 20000))
validation_patience = int(config.get('validation_patience', 5))
loss = config.get('loss', "MSE")
model = config.get('model', "mesh_scnn")
gpu_id = int(config.get('gpu_id', 0))
seed = int(config.get('seed', 42))
mesh_subdivide = int(config.get('mesh_subdivide', 1))
kappa = float(config.get('kappa', 100))
n_fibers = config.get('n_fibers', 'both')
healpix = bool(config.get('healpix', True))
min_separation_angle = int(config.get('min_separation_angle', 0))
save_dir = config.get('save_dir', "./models")
test_dir = config.get('test_dir', "./test_data")
project_name = config.get('project_name', "deepfixel")

datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = f"deepfixel_{model}_{datetime_str}"

deep_fixel.train_mesh_model(
    project_name=project_name,
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
)