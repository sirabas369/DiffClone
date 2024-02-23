# DiffClone: Enhanced Behaviour Cloning in Robotics with Diffusion-Driven Policy Learning
Sabariswaran Mani, Abhranil Chandra, Sreyas Venkataraman, Adyan Rizvi, Yash Sirvi, Soumojit Bhattacharya, Aritra Hazra

### **Winning** Solution at TOTO Competition, NeurIPS 2023

### [Project Page](https://sites.google.com/view/iitkgp-nips23toto/home) | [Paper](https://arxiv.org/abs/2401.09243)

This is the official PyTorch implementation of "DiffClone: Enhanced Behaviour Cloning in Robotics with Diffusion-Driven Policy Learning", Top Leaderboard submmission at [Train Offline Test Online Workshop Competition(TOTO)](https://toto-benchmark.org/) at NeurIPS 2023. This repo is built on the competition kit at [repo](https://github.com/AGI-Labs/toto_benchmark).

Abstract: *Robot learning tasks are extremely compute-intensive and hardware-specific. Thus the avenues of tackling these challenges, using a diverse dataset of offline demonstrations that can be used to train robot manipulation agents, is very appealing. The Train-Offline-Test-Online (TOTO) Benchmark provides a well-curated open-source dataset for offline training comprised mostly of expert data and also benchmark scores of the common offline-RL and behaviour cloning agents. In this paper, we introduce DiffClone, an offline algorithm of enhanced behaviour cloning agent with diffusion-based policy learning, and measured the efficacy of our method on real online physical robots at test time. This is also our official submission to the Train-Offline-Test-Online (TOTO) Benchmark Challenge organized at NeurIPS 2023. We experimented with both pre-trained visual representation and agent policies. In our experiments, we find that MOCO finetuned ResNet50 performs the best in comparison to other finetuned representations. Goal state conditioning and mapping to transitions resulted in a minute increase in the success rate and mean-reward. As for the agent policy, we developed DiffClone, a behaviour cloning agent improved using conditional diffusion.*

## BibTex
```bibtex
@inproceedings{
mani2024diffclone,
title={DiffClone: Enhanced Behaviour Cloning in Robotics with Diffusion-Driven Policy Learning},
author={Sabariswaran Mani and Abhranil Chandra and Sreyas Venkataraman and Adyan Rizvi and Yash Sirvi and Soumojit Bhattacharya and Aritra Hazra},
journal={arXiv preprint arXiv:2401.09243},
year={2024}
}
```

## Prerequisites
- Conda

## Installation
You can either use a local conda environment or a docker environment.

### Setup conda environment
1. Run the following command to create a new conda environment: ```source setup_toto_env.sh```

### Download vision representation models
Vision representation models that were used [here](https://drive.google.com/drive/folders/1iqDIIIalTi3PhAnFjZxesksvFVldK42p?usp=sharing). This contains the in-domain models (MoCo and BYOL) that are trained with the TOTO dataset.

## Simulator installation
Pouring simulator uses DeepMind MuJoCo, which you can install with this command:
  ```
  pip install mujoco
  ```
To set up MuJoCo rendering, install egl following the instructions [here](https://pytorch.org/rl/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html#prerequisite-for-rendering-all-mujoco-versions).

You can check that the environment is properly installed by running the following from the toto_benchmark directory:
  ```
  (toto-benchmark) user@machine:~$ MUJOCO_GL=egl python
  Python 3.8.0 | packaged by conda-forge | (default, Nov 22 2019, 19:11:38)
  [GCC 7.3.0] :: Anaconda, Inc. on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> from toto_benchmark.sim.dm_pour import DMWaterPouringEnv
  >>> eval_env = DMWaterPouringEnv()
  ```

## Download simulation dataset
The simulation dataset can be downloaded [here](https://drive.google.com/drive/folders/1HKtjLBgI6FJlMj44Tbr_cDPUCCz-mEZO?usp=sharing). The file contains 103 human-teleoperated trajectories of the pouring task.

## Train a DiffClone Agent
Here's an example command to train an image-based DiffClone agent on the simulation data with MOCO as the image encoder. Our agent assumes that each image has been encoded into a 1D vector. You will need to download dm_human_dataset.pickle to have this launched. 
```
cd toto_benchmark
 
python scripts/train.py --config-name train_diffusion_unet_sim.yaml 
```

The config train_diffusion_unet_sim.yaml is set up to train our DiffClone agent on the simulated pouring task. Before launching the command, [download the simulation dataset](#download-simulation-dataset) and put the file under `toto_benchmark/sim`.

## Simulation evaluation
To evaluate an agent following the Diffusion training, run the following command:
  ```
python toto_benchmark/sim/eval_agent.py -f outputs/<path_to>/<agent>/
  ```

## Real-world Training and Evaluation
The final submission to the competition were on Real-world datasets. Also the code to train and evaluate on this setting is also present in our codebase. Just use train_diffusion.yaml as config. It automatically use the right code. Please refer to the original TOTO Repository for further detailed instructions and to download the real-world datasets.

## Acknowledgement
The code is built on [TOTO](https://github.com/AGI-Labs/toto_benchmark). We thank the organizers of [TOTO](https://toto-benchmark.org/) Competition for the opportunity to participate in the competition and contribute to the field.
