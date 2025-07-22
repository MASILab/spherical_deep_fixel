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
save_dir = "../models"
test_dir = "../test_data"

name = "deepfixel_mesh_scnn_healpix_2025-04-14_12-21-03"

output_dir_orig = f'../outputs/peak_finding_orig'
amp_threshold = 0.1
model_path = f"../models/{name}/best_model.pth"

start_time = time.time()

deep_fixel.test_mesh_model(
    model=model,
    model_path=model_path,
    batch_size=batch_size,
    n_fibers=n_fibers,
    subdivide_mesh=mesh_subdivide,
    amp_threshold=amp_threshold,
    output_dir=output_dir_orig,
    kappa=kappa,
    test_dir=test_dir,
    gpu_id=gpu_id,
    healpix=healpix,
    use_dipy=False
)

orig_time = time.time() - start_time

output_dir_dipy = f'../outputs/peak_finding_dipy'

start_time = time.time()

deep_fixel.test_mesh_model(
    model=model,
    model_path=model_path,
    batch_size=batch_size,
    n_fibers=n_fibers,
    subdivide_mesh=mesh_subdivide,
    amp_threshold=0,
    output_dir=output_dir_dipy,
    kappa=kappa,
    test_dir=test_dir,
    gpu_id=gpu_id,
    healpix=healpix,
    use_dipy=True,
    min_separation_angle=0,
    is_symmetric=True
)

dipy_time = time.time() - start_time

output_dir_dipy = f'../outputs/peak_finding_dipy_nl'

start_time = time.time()

deep_fixel.test_mesh_model(
    model=model,
    model_path=model_path,
    batch_size=batch_size,
    n_fibers=n_fibers,
    subdivide_mesh=mesh_subdivide,
    amp_threshold=0,
    output_dir=output_dir_dipy,
    kappa=kappa,
    test_dir=test_dir,
    gpu_id=gpu_id,
    healpix=healpix,
    use_dipy="nl",
    min_separation_angle=0,
)

dipy_nl_time = time.time() - start_time

# Load results
orig_results = pd.read_csv(f"{output_dir_orig}/test_results.csv")
dipy_results = pd.read_csv(f"{output_dir_dipy}/test_results.csv")
dipy_nl_results = pd.read_csv(f"{output_dir_dipy}/test_results.csv")

# Combine with a new column for peak_finding_method
orig_results['peak_finding_method'] = 'original'
dipy_results['peak_finding_method'] = 'dipy'
dipy_nl_results['peak_finding_method'] = 'dipy_nl'

combined_results = pd.concat([orig_results, dipy_results, dipy_nl_results], ignore_index=True)

# Plot boxplot with hue for peak_finding_method
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=combined_results, x='peak_finding_method', y='acc', ax=ax)
sns.stripplot(data=combined_results, x='peak_finding_method', y='acc', ax=ax, color='black', alpha=0.5, jitter=True)
ax.set_title('Comparison of Peak Finding Methods')
ax.set_ylabel('ACC')
ax.set_xlabel('Peak Finding Method')

pairs = [('original', 'dipy'), ('original', 'dipy_nl')]
annotator = Annotator(ax, pairs, data=combined_results, x='peak_finding_method', y='acc')
annotator.configure(test='Wilcoxon', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

# Print median and IQR
print(combined_results.groupby('peak_finding_method')['acc'].aggregate(['median', lambda x: x.quantile(0.75) - x.quantile(0.25)]))

print(f"Original peak finding time: {orig_time:.2f} seconds")
print(f"Dipy peak finding time: {dipy_time:.2f} seconds")
print(f"Dipy NL peak finding time: {dipy_nl_time:.2f} seconds")

plt.show()