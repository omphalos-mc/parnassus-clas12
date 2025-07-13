#!/bin/bash
##PBS -l gputype=A6000
IOTHROTTLE_LIMIT=0

export IOTHROTTLE_VERBOSE=1
source /usr/wipp/conda/24.5.0/bin/activate /usr/wipp/conda/24.5.0u/envs/common
cd /srv01/agrp/dmitrykl/projects/parnassus-core/src/experimentscms_2011

# python train.py -c configs/fm_particle.yaml --gpus 0
# python train.py -c configs/fm_event.yaml --gpus 0
python train.py -c configs/fm_impact.yaml --gpus 0