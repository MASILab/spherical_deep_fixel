FROM python:3.12-slim-trixie

RUN apt-get update && apt-get install -y git

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Need to add hsd as a package, but original GitHub does not have pyproject.toml
# Use this as a workaround
RUN mkdir -p /opt/hsd/src/hsd/
RUN git clone https://github.com/AxelElaldi/fast-equivariant-deconv /opt/hsd/src/hsd/
RUN touch /opt/hsd/src/hsd/__init__.py /opt/hsd/src/hsd/utils/__init__.py /opt/hsd/src/hsd/model/__init__.py
RUN cat > /opt/hsd/pyproject.toml <<'PYPROJECT'
[project]
name = "hsd"
version = "0.1.0"
description = "https://github.com/AxelElaldi/fast-equivariant-deconv"
readme = "README.md"
dependencies = [
    "h5py==3.11.0",
    "healpy==1.16.6",
    "joblib==1.4.2",
    "matplotlib==3.9.0",
    "nibabel==5.2.1",
    "numpy<2",
    "pandas==2.2.2",
    "pygsp==0.5.1",
    "pyyaml==6.0.1",
    "scipy==1.13.0",
    "tensorboard==2.16.2",
    "torch==2.3.0",
    "torchaudio==2.3.0",
    "torchvision==0.18.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
PYPROJECT

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /opt/hsd

RUN mkdir -p /app/models
ADD https://zenodo.org/records/17859792/files/best_model_mlp.pth?download=1 /app/models/best_model_mlp.pth
ADD https://zenodo.org/records/17859792/files/best_model_scnn.pth?download=1 /app/models/best_model_scnn.pth

ENV PATH="/app/.venv/bin:${PATH}"