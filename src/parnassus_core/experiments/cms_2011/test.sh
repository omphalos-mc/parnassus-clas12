#!/bin/bash
TOP_DIR=/srv01/agrp/dmitrykl/projects/parnassus-core/src/parnassus_core/experiments/cms_2011/
DATA_DIR=/storage/agrp/dmitrykl/fastsim/f_delphes/data/full_event/output_impact/

TTBAR_TEST=$DATA_DIR"TTJets_MSDecays_central_TuneZ2_7TeV-madgraph-tauola/merged_test_v2_pt025_eta27_ghost.root"

# Event models
# python main.py test \
#     -c $TOP_DIR"logs/cms2011_event_fm_MSE_20250816-T151325/config.yaml" \
#     --ckpt_path $TOP_DIR"logs/cms2011_event_fm_MSE_20250816-T151325/ckpts/epoch=295-val_loss=3.44413.ckpt" \
#     --data.test_path $TTBAR_TEST \
#     --data.num_test 500 --data.eval_output_name ttbar

# Particle models
# python main.py test \
#     -c $TOP_DIR"logs/cms2011_particle_rcfm_MSEDir_Deep5M_Flex_20250716-T221618/config.yaml" \
#     --ckpt_path $TOP_DIR"logs/cms2011_particle_rcfm_MSEDir_Deep5M_Flex_20250716-T221618/ckpts/epoch=105-val_loss=0.72356.ckpt" \
#     --data.test_path $TTBAR_TEST \
#     --data.num_test 500 --data.eval_output_name ttbar

# Impact models
# python main.py test \
#     -c $TOP_DIR"logs/cms2011_impact_icfm_MSEDir_FlashSim_20250725-T165102/config.yaml" \
#     --ckpt_path $TOP_DIR"logs/cms2011_impact_icfm_MSEDir_FlashSim_20250725-T165102/ckpts/epoch=054-val_loss=1.03487.ckpt" \
#     --data.test_path $TTBAR_TEST \
#     --data.num_test 500 --data.eval_output_name ttbar
