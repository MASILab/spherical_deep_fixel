# DeepFixel: Crossing white matter fiber identification through spherical convolutional neural networks
[![arXiv](https://img.shields.io/badge/arXiv-2511.03893-b31b1b.svg)](https://arxiv.org/abs/2511.03893)

DeepFixel is a deep learning method for identification of crossing fiber bundle elements from diffusion MRI.

> Adam M. Saunders, Lucas W. Remedios, Elyssa M. McMaster, Jongyeon Yoon, Gaurav Rudravaram, Adam Sadriddinov, Praitayini Kanakaraj, Bennett A, Landman, and Adam W. Anderson. DeepFixel: Crossing white matter fiber identification through spherical convolutional neural networks. SPIE Medical Imaging: Clinical and Biomedical Imaging, 2026. [https://arxiv.org/abs/2511.03893](https://arxiv.org/abs/2511.03893)

## Installation
You can set up an environment using [`uv`](https://github.com/astral-sh/uv) by running the following command:
```bash
uv sync
```

Alternatively, you can use Docker or Apptainer (see instructions below).

## Usage
To run the model, download the weights and testing dataset from the following link: [https://zenodo.org/records/16587458](https://zenodo.org/records/16587458). 
- Unzip and copy the testing data to `./test_data`
- Put the weights in `./models/pretrained`


To train the model:
```bash
python train_deep_fixel.py --config config/example_scnn.yaml
```

To test the model on the provided testing dataset:
```bash
python test_deep_fixel.py --config config/example_scnn.yaml
```


## Usage (Docker)
To build the Docker image, clone the repository and run the following command in the root directory:
```bash
sudo docker build -t spherical_deep_fixel:v1.0.0 .
```

Then run the Docker container with the following command (note you will likely need to bind in local directories with `-v`):
```bash
sudo docker run --rm -it --gpus all -v $(pwd):$(pwd) $spherical_deep_fixel:v1.0.0 python train_deep_fixel.py --config /path/to/config/example_scnn.yaml
sudo docker run --rm -it --gpus all -v $(pwd):$(pwd) $spherical_deep_fixel:v1.0.0 python test_deep_fixel.py --config /path/to/config/example_scnn.yaml
```

## Usage (Apptainer)
A pre-built Apptainer image is available on Zenodo: 

```bash
apptainer run -C -B $(pwd):$(pwd) --nv https://zenodo.org/records/16587458/files/spherical_deep_fixel_v1.0.0.sif python /app/train_deep_fixel.py --config /path/to/config/example_scnn.yaml
apptainer run -C -B $(pwd):$(pwd) --nv https://zenodo.org/records/16587458/files/spherical_deep_fixel_v1.0.0.sif python /app/test_deep_fixel.py --config /path/to/config/example_scnn.yaml
```

## Applying the model to your own data
If you wish to apply the model to your own dataset, you can use `fissile.test_mesh_model()` as a basis for your code. You can also use `fissile.dataset.GeneratedMeshNIFTIDataset()` if your data is stored as spherical harmonic coefficients in a NIFTI file.

## Citation
If you use this code in your research, please cite the following paper:

> Adam M. Saunders, Lucas W. Remedios, Elyssa M. McMaster, Jongyeon Yoon, Gaurav Rudravaram, Adam Sadriddinov, Praitayini Kanakaraj, Bennett A, Landman, and Adam W. Anderson. DeepFixel: Crossing white matter fiber identification through spherical convolutional neural networks. SPIE Medical Imaging: Clinical and Biomedical Imaging, 2026. [https://arxiv.org/abs/2511.03893](https://arxiv.org/abs/2511.03893)