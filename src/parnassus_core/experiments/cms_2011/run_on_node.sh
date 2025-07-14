#!/bin/bash
##PBS -l gputype=A6000
IOTHROTTLE_LIMIT=0
export IOTHROTTLE_LIMIT=0
export IOTHROTTLE_VERBOSE=1
source /usr/wipp/conda/24.5.0/bin/activate /usr/wipp/conda/24.5.0u/envs/common
cd /srv01/agrp/dmitrykl/projects/parnassus-core/src/parnassus_core/experiments/cms_2011

# python main.py fit -c configs/particle.yaml
# python main.py fit -c configs/event.yaml
python main.py fit -c configs/impact.yaml