This repository contains the code associated with our WACV 2024 paper ([arxiv](https://arxiv.org/abs/2311.02872)).
A large part of the code is built upon the [ACE](https://github.com/nianticlabs/ace) code.

## Installation
```shell
conda env create -f environment.yml
conda activate focus_tune
conda install -c conda-forge pykdtree tqdm
```
or if you use [Mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html).
```shell
mamba env create -f environment.yml
conda activate focus_tune
mamba install -c conda-forge pykdtree tqdm
```

**All the following commands in this file need to run in the `focus_tune` environment.**

## Datasets

### 7 scenes:
To download and prepare the datasets using the PGT poses:

```shell
cd datasets
# Downloads the data to datasets/pgt_7scenes_{chess, fire, ...}
./setup_7scenes.py --poses pgt
``` 

3D models must be downloaded from [here](https://github.com/tsattler/visloc_pseudo_gt_limitations), unzip them to the parent directory of this repo.

### Cambridge Landmarks:
To download the dataset along with the 3D models:
```shell
cd datasets
# Downloads the data to datasets/Cambridge_{GreatCourt, KingsCollege, ...}
./setup_cambridge.py
```

## Usage
### ACE Training

The ACE scene-specific coordinate regression head for a scene can be trained using the `train_ace.py` script.
Basic usage:

```shell
./train_ace.py <scene path> <output map name> --constraint_mask 1 --sampling_radius 5
# Example:
./train_ace.py datasets/pgt_7scenes_chess output/7scenes_chess.pt --constraint_mask 1 --sampling_radius 5
```

### ACE Evaluation

The pose estimation for a testing scene can be performed using the `test_ace.py` script.
Basic usage:

```shell
./test_ace.py <scene path> <output map name>
# Example:
./test_ace.py datasets/7scenes_chess output/7scenes_chess.pt
```

#### Ensemble Evaluation
ACE Poker example for a scene in the Cambridge dataset:

```shell
./run_poker.sh
```
or
```shell
mkdir -p output/Cambridge_GreatCourt

# Head training:
./train_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/0_4.pt --num_clusters 4 --cluster_idx 0
./train_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/1_4.pt --num_clusters 4 --cluster_idx 1
./train_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/2_4.pt --num_clusters 4 --cluster_idx 2
./train_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/3_4.pt --num_clusters 4 --cluster_idx 3

# Per-cluster evaluation:
./test_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/0_4.pt --session 0_4
./test_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/1_4.pt --session 1_4
./test_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/2_4.pt --session 2_4
./test_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/3_4.pt --session 3_4

# Merging results and computing metrics.

# The merging script takes a --poses_suffix argument that's used to select only the 
# poses generated for the requested number of clusters. 
./merge_ensemble_results.py output/Cambridge_GreatCourt output/Cambridge_GreatCourt/merged_poses_4.txt --poses_suffix "_4.txt"

# The output poses output by the previous script are then evaluated against the scene ground truth data.
./eval_poses.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/merged_poses_4.txt
```
