# DeepFixel
Deep learning-based identification of crossing fiber bundle elements

## Training and testing the model
You can set up an environment using [`uv`](https://github.com/astral-sh/uv) by running the following command:
```bash
uv sync
```

To run the model, download the weights and testing dataset from the following link: [DeepFixel](https://zenodo.org/records/14962758).

- Unzip and copy the testing data to `./test_data`
- Put the weights in `./models/pretrained`

See `run_pretrained_deep_fixel.py` to test the pretrained model and `run_deep_fixel.py` to train and test a new model.

## Using the model
If you wish to apply the model to your own dataset, you can use `fissile.test_mesh_model()` as a basis for your code. You can also use `fissile.dataset.GeneratedMeshNIFTIDataset()` if your data is stored as spherical harmonic coefficients in a NIFTI file.