#!/bin/bash
##PBS -l gputype=A6000
source /usr/wipp/conda/24.5.0/bin/activate /usr/wipp/conda/24.5.0u/envs/common
cd /srv01/agrp/dmitrykl/projects/parnassus-core/experiments/cms_2011

# python train.py -c configs/fm_particle.yaml --gpus 0
# python train.py -c configs/fm_event.yaml --gpus 0
python train.py -c configs/fm_impact.yaml --gpus 0