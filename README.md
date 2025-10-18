# HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos

<div align="center">
<a href="https://hdmi-humanoid.github.io/">
  <img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://www.youtube.com/watch?v=GvIBzM7ieaA&list=PL0WMh2z6WXob0roqIb-AG6w7nQpCHyR0Z&index=12">
  <img alt="Video" src="https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=youtube"/>
</a>

<a href="https://arxiv.org/pdf/2509.16757">
  <img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://github.com/LeCAR-Lab/HDMI/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/LeCAR-Lab/HDMI?style=social"/>
</a>


</div>

HDMI is a novel framework that enables humanoid robots to acquire diverse whole-body interaction skills directly from monocular RGB videos of human demonstrations.

This repository contains the official training code of **HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos**.


## TODO
- [x] Release hdmi training code 
- [x] hoi motion datasets
- [x] Release pretrained models
- [x] Release sim2real code


## 🚀 Quick Start

```bash
# setup conda environment
conda create -n hdmi python=3.11 -y
conda activate hdmi

# install mjlab (my fork)
git clone git@github.com:EGalahad/mjlab.git
cd mjlab
pip install -e .

# install hdmi
cd ..
git clone https://github.com/EGalahad/hdmi
cd hdmi
git checkout mjlab
pip install -e .

```

Alternative way (`uv`)

```bash
uv sync
```

## Train and Evaluate

Teacher policy 
```bash
# train policy
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
# evaluate policy
python scripts/play.py algo=ppo_roa_train task=G1/hdmi/move_suitcase checkpoint_path=run:<wandb-run-path>

# or use uv run
uv run scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
```

Student policy
```bash
# train policy
python scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<teacher_wandb-run-path>
# evaluate policy
python scripts/play.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<student_wandb-run-path>
```
## Sim2Real

Please see [github.com/EGalahad/sim2real](https://github.com/EGalahad/sim2real) for details.
