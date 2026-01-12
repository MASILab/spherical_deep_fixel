# DeepFixel: Crossing white matter fiber identification through spherical convolutional neural networks
[![arXiv](https://img.shields.io/badge/arXiv-2511.03893-b31b1b.svg)](https://arxiv.org/abs/2511.03893) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17834289.svg)](https://doi.org/10.5281/zenodo.17834289)

DeepFixel is a deep learning method for identification of crossing fiber bundle elements from diffusion MRI.

> Adam M. Saunders, Lucas W. Remedios, Elyssa M. McMaster, Jongyeon Yoon, Gaurav Rudravaram, Adam Sadriddinov, Praitayini Kanakaraj, Bennett A, Landman, and Adam W. Anderson. DeepFixel: Crossing white matter fiber identification through spherical convolutional neural networks. SPIE Medical Imaging: Clinical and Biomedical Imaging, 2026. [https://arxiv.org/abs/2511.03893](https://arxiv.org/abs/2511.03893)

## Installation
You can set up an environment using [`uv`](https://github.com/astral-sh/uv) by running the following command:
```bash
uv sync
```

Alternatively, you can use Docker or Apptainer (see instructions below).

## Usage
You can use DeepFixel to split multi-fiber ODFs into the underlying single-fiber ODFs. The pretrained weights are available on Zenodo: [https://doi.org/10.5281/zenodo.17834289](https://doi.org/10.5281/zenodo.17834289). 

```bash
# For pretrained models: --lmax 6, --subdivide 1
deepfixel /path/to/input/fod.nii.gz \
    /path/to/output_dir \
    /path/to/best_model_scnn.pth \
    --mask /path/to/mask.nii.gz \
    --maxnum 2 \
    --lmax 6 \
    --subdivide 1 \
    --amp_threshold 0.1 \
    --model mesh_scnn \
    --batch_size 512 \
    --gpu_id 1
```

## Training and testing the model
To run the model, download the weights and testing dataset from the following link: [https://doi.org/10.5281/zenodo.17834289](https://doi.org/10.5281/zenodo.17834289). 
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


### Docker
To build the Docker image, clone the repository and run the following command in the root directory:
```bash
sudo docker build -t spherical_deep_fixel:v1.2.0 .
```

Then run the Docker container with the following command (note you will likely need to bind in local directories with `-v`):

```bash
sudo docker run --rm -it --gpus all \
    /path/to/input/fod.nii.gz \
    /path/to/output_dir \
    /app/models/best_model_scnn.pth \
    --mask /path/to/mask.nii.gz \
    --maxnum 2 \
    --lmax 6 \
    --subdivide 1 \
    --amp_threshold 0.1 \
    --model mesh_scnn \
    --batch_size 512 \
    --gpu_id 0
```

For training and testing:
```bash
sudo docker run --rm -it --gpus all $spherical_deep_fixel:v1.2.0 python train_deep_fixel.py --config /path/to/config/example_scnn.yaml
sudo docker run --rm -it --gpus all $spherical_deep_fixel:v1.2.0 python test_deep_fixel.py --config /path/to/config/example_scnn.yaml
```

### Apptainer
A pre-built Apptainer image is available on Zenodo ([https://doi.org/10.5281/zenodo.17834289](https://doi.org/10.5281/zenodo.17834289)). (Note you will likely need to bind in local directories with `-B`):

```bash
apptainer run -C --nv spherical_deep_fixel_v1.2.0.sif \
    /path/to/input/fod.nii.gz \
    /path/to/output_dir \
    /app/models/best_model_scnn.pth \
    --mask /path/to/mask.nii.gz \
    --maxnum 2 \
    --lmax 6 \
    --subdivide 1 \
    --amp_threshold 0.1 \
    --model mesh_scnn \
    --batch_size 512 \
    --gpu_id 0
```

For training and testing:
```bash
apptainer run -C --nv spherical_deep_fixel_v1.2.0.sif python /app/train_deep_fixel.py --config /path/to/config/example_scnn.yaml
apptainer run -C --nv spherical_deep_fixel_v1.2.0.sif python /app/test_deep_fixel.py --config /path/to/config/example_scnn.yaml
```

## Citation
If you use this code in your research, please cite the following paper:

> Adam M. Saunders, Lucas W. Remedios, Elyssa M. McMaster, Jongyeon Yoon, Gaurav Rudravaram, Adam Sadriddinov, Praitayini Kanakaraj, Bennett A, Landman, and Adam W. Anderson. DeepFixel: Crossing white matter fiber identification through spherical convolutional neural networks. SPIE Medical Imaging: Clinical and Biomedical Imaging, 2026. [https://arxiv.org/abs/2511.03893](https://arxiv.org/abs/2511.03893)