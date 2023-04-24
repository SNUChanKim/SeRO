# SeRO: Self-Supervised Reinforcement Learning for Recovery from Out-of-Distribution Situations
Pytorch code for the submission:
>SeRO: Self-Supervised Reinforcement Learning for Recovery from Out-of-Distribution Situations, IJCAI 2023

## Requirements
- mujoco200 (https://www.roboti.us/)


## Generating and activating conda environment
```
$ conda env create -f env.yml
$ conda activate sero
```

## Training phase
### HalfCheetahNormal-v2
```
$ cd ~/directory/to/repository/
$ . scripts/train_{algo}_half_cheetah_normal.sh
```
### HopperNormal-v2
```
$ cd ~/directory/to/repository/
$ . scripts/train_{algo}_hopper_normal.sh
```
### Walker2DNormal-v2
```
$ cd ~/directory/to/repository/
$ . scripts/train_{algo}_walker2d_normal.sh
```
### AntNormal-v2
```
$ cd ~/directory/to/repository/
$ . scripts/train_{algo}_ant_normal.sh
```
## Retraining phase (should be executed after the training phase)
### HalfCheetahOOD-v2
```
$ cd ~/directory/to/repository/
$ . scripts/train_{algo}_half_cheetah_ood.sh
```
### HopperOOD-v2
```
$ cd ~/directory/to/repository/
$ . scripts/train_{algo}_hopper_ood.sh
```
### Walker2DOOD-v2
```
$ cd ~/directory/to/repository/
$ . scripts/train_{algo}_walker2d_ood.sh
```
### AntOOD-v2
```
$ cd ~/directory/to/repository/
$ . scripts/train_{algo}_ant_ood.sh
```

## Visualize learning curves
```
$ cd ~/directory/to/repository/log/
$ tensorboard --logdir={env_name}
```