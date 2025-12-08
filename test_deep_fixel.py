import deep_fixel
from datetime import datetime
from argparse import ArgumentParser
import yaml

parser = ArgumentParser(description="Test DeepFixel model")
parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

lr = float(config.get('lr', 1e-3))
batch_size = int(config.get('batch_size', 512))
loss = config.get('loss', "MSE")
model = config.get('model', "mesh_scnn")
gpu_id = int(config.get('gpu_id', 0))
seed = int(config.get('seed', 42))
mesh_subdivide = int(config.get('mesh_subdivide', 1))
healpix = bool(config.get('healpix', True))
kappa = float(config.get('kappa', 100))
min_separation_angle = int(config.get('min_separation_angle', 0))
n_fibers = config.get('n_fibers', 'both')
model_path = config.get('pretrained_model_path', "./models/pretrained/best_model_scnn.pth")
test_dir = config.get('test_dir', "./test_data")
output_dir = config.get('output_dir', './outputs/pretrained_scnn')
amp_threshold = float(config.get('amp_threshold', 0.1))

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
