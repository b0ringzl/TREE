# pctrees LiDAR tree species classification deployment

## Location

- Local clone: `D:\TREE\Lidar\pctrees_github`
- Source repository: `https://github.com/mattynaz/pctrees.git`
- pctrees commit at clone time: `c19c643 clean scripts`
- PCT_Pytorch submodule used locally: `0199a3b Update LICENSE`

## Conda Environment

Deployment uses the existing `yolo` conda environment:

```powershell
conda activate yolo
```

The following missing Python packages were installed into `yolo`:

```text
torchmetrics
imageio
laspy
transforms3d
geopandas
wandb
pointnet2_ops
```

Existing PyTorch was preserved:

```text
torch 2.11.0.dev20260210+cu128
CUDA 12.8
GPU NVIDIA GeForce RTX 5060 Laptop GPU, compute capability 12.0
```

## Local Deployment Fixes

The upstream pctrees repository currently records a PCT_Pytorch submodule commit that is no longer fetchable from `https://github.com/Strawberry-Eat-Mango/PCT_Pytorch.git`:

```text
3b2aaabb9514028d83a0af7d571a1d71c7a4be7a
```

For local deployment, the available current PCT_Pytorch main branch was checked out instead:

```text
0199a3b8c89274aeeebb5634c82a47573c8be492
```

The following local patches were applied inside `pctrees_github`:

- `train.py`: default W&B mode is offline so `python main.py --help` does not require a W&B API key.
- `pct_main.py`: adds `PCT_Pytorch` to `sys.path`, uses Windows-compatible backup copying, fixes default `model_path`, and makes `train_split` a float.
- `PCT_Pytorch/pointnet2_ops_lib/setup.py`: defaults CUDA architecture to `12.0` and passes `-allow-unsupported-compiler` for the installed Visual Studio Build Tools/CUDA combination.

## Start Commands

Recommended LiDAR training entry:

```powershell
D:\TREE\Lidar\train_pctrees_lidar.bat
```

The first run builds a reusable point tensor cache under:

```text
D:\TREE\lidar data\pctrees_cache_4096
```

After the cache exists, later training runs read `.pt` tensors instead of repeatedly
decompressing `.laz` files.

To choose an experiment name and epoch count:

```powershell
D:\TREE\Lidar\train_pctrees_lidar.bat tree_lidar_pct_run1 10
```

To choose cache size and DataLoader worker count:

```powershell
D:\TREE\Lidar\train_pctrees_lidar.bat tree_lidar_pct_run1 10 4096 2
```

If Windows multiprocessing is unstable on a run, use worker count `0`:

```powershell
D:\TREE\Lidar\train_pctrees_lidar.bat tree_lidar_pct_run1 10 4096 0
```

To rebuild the cache manually:

```powershell
python D:\TREE\Lidar\build_pctrees_point_cache.py --cache-points 4096 --overwrite
```

Watch live progress in another terminal:

```powershell
powershell -ExecutionPolicy Bypass -File D:\TREE\Lidar\watch_pctrees_training.ps1
```

Baseline/SimpleView entry:

```powershell
D:\TREE\Lidar\run_pctrees_baseline.bat --help
```

PCT/PCTreeS entry:

```powershell
D:\TREE\Lidar\run_pctrees_pct.bat --help
```

Training requires a LiDAR data directory containing tree scan files and a compatible `labels.csv`.

## Verified

The following checks passed in `yolo`:

```powershell
python main.py --help
python pct_main.py --help
python -c "import pointnet2_ops; from PCT_Pytorch.model import Pct"
```

No real training run was executed yet because project-specific Hong Kong tree LiDAR data has not been mapped into the pctrees dataset format.
