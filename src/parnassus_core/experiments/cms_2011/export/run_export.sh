#/bin/bash

echo "Running export script..."

echo "Exporting Event models"

python export.py \
    --ckpt ../logs/cms2011_event_fm_MSE_20250825-T154555/ckpts/last.ckpt \
    --config ../logs/cms2011_event_fm_MSE_20250825-T154555/config.yaml \
    --output evt.onnx \
    --mode onnx

python export.py \
    --ckpt ../logs/cms2011_event_fm_MSE_20250825-T154555/ckpts/last.ckpt \
    --config ../logs/cms2011_event_fm_MSE_20250825-T154555/config.yaml \
    --output evt.pt2 \
    --mode torch

echo "Exporting Particle models"

python export.py \
    --ckpt ../logs/cms2011_particle_rcfm_MSEDir_Deep5M_Flex_Simple_20250825-T161212/ckpts/last.ckpt \
    --config ../logs/cms2011_particle_rcfm_MSEDir_Deep5M_Flex_Simple_20250825-T161212/config.yaml \
    --output particle.onnx \
    --mode onnx

python export.py \
    --ckpt ../logs/cms2011_particle_rcfm_MSEDir_Deep5M_Flex_Simple_20250825-T161212/ckpts/last.ckpt \
    --config ../logs/cms2011_particle_rcfm_MSEDir_Deep5M_Flex_Simple_20250825-T161212/config.yaml \
    --output particle.pt2 \
    --mode torch

echo "Exporting Impact models"

python export.py \
    --ckpt ../logs/cms2011_impact_fm_MSEDir_FlashSim_20250825-T162621/ckpts/last.ckpt \
    --config ../logs/cms2011_impact_fm_MSEDir_FlashSim_20250825-T162621/config.yaml \
    --output impact.onnx \
    --mode onnx

python export.py \
    --ckpt ../logs/cms2011_impact_fm_MSEDir_FlashSim_20250825-T162621/ckpts/last.ckpt \
    --config ../logs/cms2011_impact_fm_MSEDir_FlashSim_20250825-T162621/config.yaml \
    --output impact.pt2 \
    --mode torch
