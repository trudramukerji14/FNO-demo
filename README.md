# FNO-demo

A demonstration of aeromagnetic inversion using neural operators and Bayesian optimization.

## Overview

This project implements a full pipeline for recovering subsurface magnetic susceptibility parameters from aeromagnetic survey data:

1. **GP Prior** — Sample realistic 3D susceptibility fields using a composite Matérn Gaussian process
2. **Forward Simulation** — Compute 2D aeromagnetic anomaly maps using SimPEG
3. **FNO Surrogate** — Train a Fourier Neural Operator to replace the expensive SimPEG forward model
4. **Bayesian Inversion** — Use BoTorch to recover subsurface parameters from observed anomaly data

## Attribution

Sections 1–3 of `demo.ipynb` are based on [williamjsdavis/GP-cubed](https://github.com/williamjsdavis/GP-cubed) (MIT License), specifically the [`neural-operator-geophysics`](https://github.com/williamjsdavis/GP-cubed/tree/main/neural-operator-geophysics) subdirectory. The `src.py` file is copied directly from that repository. Section 4 (Bayesian inversion with BoTorch) is the contribution here.
## Requirements

Dependencies are managed with [uv](https://github.com/astral-sh/uv). To install:

```bash
uv sync
```

Key dependencies: `torch`, `gpytorch`, `botorch`, `neuralop`, `simpeg`, `shapely`

## Usage

Open `demo.ipynb` in VSCode or JupyterLab and run cells in order.

To skip dataset generation (slow), load the pre-saved dataset:
```python
dataset = torch.load("aeromag_dataset.pt", weights_only=False)
```

To load pre-trained FNO weights:
```python
fno.load_state_dict(torch.load("fno_weights.pt", weights_only=True))
```

## Notes

- Optimized for Apple Silicon (MPS). Falls back to CPU automatically on other hardware.
- SimPEG forward simulation is CPU-only; dataset generation does not use MPS.
