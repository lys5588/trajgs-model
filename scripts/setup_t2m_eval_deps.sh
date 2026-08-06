#!/bin/bash
# Setup T2M Evaluation Dependencies
# Checks for the dependency files required for T2M/M2T evaluation and prints
# instructions for the ones that are missing. All assets come from the official
# HumanML3D repository (https://github.com/EricGuo5513/HumanML3D) and must be
# downloaded manually (they are not redistributed here).
#
# Expected layout after setup:
#   1. T2M evaluator weights:
#        deps/t2m/t2m/text_mot_match/model/finest.tar
#        deps/t2m/t2m/Comp_v6_KLD01/meta/mean.npy
#        deps/t2m/t2m/Comp_v6_KLD01/meta/std.npy
#   2. Processed GloVe vocabulary (HumanML3D format):
#        deps/glove/glove/our_vab_data.npy
#        deps/glove/glove/our_vab_idx.pkl
#        deps/glove/glove/our_vab_words.pkl
#
# Usage:
#   bash scripts/setup_t2m_eval_deps.sh

set -e

DEPS_DIR="deps"
T2M_DIR="${DEPS_DIR}/t2m"
GLOVE_DIR="${DEPS_DIR}/glove"

echo "=========================================="
echo "Setting up T2M Evaluation Dependencies"
echo "=========================================="

# Create directories
mkdir -p "$T2M_DIR"
mkdir -p "$GLOVE_DIR"

# ==================== T2M Evaluator ====================
echo ""
echo "[1/2] T2M Evaluator weights..."

T2M_CHECKPOINT="${T2M_DIR}/t2m/text_mot_match/model/finest.tar"
T2M_META_MEAN="${T2M_DIR}/t2m/Comp_v6_KLD01/meta/mean.npy"
T2M_META_STD="${T2M_DIR}/t2m/Comp_v6_KLD01/meta/std.npy"

if [ -f "$T2M_CHECKPOINT" ]; then
    echo "  ✓ T2M evaluator already exists at $T2M_CHECKPOINT"
else
    echo "  T2M evaluator weights not found."
    echo ""
    echo "  Please download the pretrained evaluator bundle from the HumanML3D"
    echo "  repository (it ships a t2m/ folder with the trained models):"
    echo "  https://github.com/EricGuo5513/HumanML3D"
    echo ""
    echo "  Place the files so that the following path exists:"
    echo "  -----------------------------------------------"
    echo "  mkdir -p ${T2M_DIR}/t2m/text_mot_match/model"
    echo "  # Put finest.tar at:"
    echo "  # ${T2M_CHECKPOINT}"
    echo "  -----------------------------------------------"
    echo ""
    echo "  Alternatively, the same evaluator is distributed with the MLD repository:"
    echo "  https://github.com/ChenFengYe/motion-latent-diffusion"
fi

if [ -f "$T2M_META_MEAN" ] && [ -f "$T2M_META_STD" ]; then
    echo "  ✓ Evaluator meta stats already exist at ${T2M_DIR}/t2m/Comp_v6_KLD01/meta/"
else
    echo "  Evaluator meta stats (mean.npy / std.npy) not found."
    echo ""
    echo "  The same HumanML3D t2m/ bundle contains Comp_v6_KLD01/meta/."
    echo "  Place the files so that the following paths exist:"
    echo "  -----------------------------------------------"
    echo "  mkdir -p ${T2M_DIR}/t2m/Comp_v6_KLD01/meta"
    echo "  # Put mean.npy and std.npy at:"
    echo "  # ${T2M_META_MEAN}"
    echo "  # ${T2M_META_STD}"
    echo "  -----------------------------------------------"
    echo ""
    echo "  Warning: without these files the evaluation falls back to the training-set"
    echo "  Mean/Std, which silently changes the reported metrics."
fi

# ==================== GloVe (HumanML3D processed vocabulary) ====================
echo ""
echo "[2/2] GloVe word embeddings (HumanML3D processed format)..."

GLOVE_VAB_DATA="${GLOVE_DIR}/glove/our_vab_data.npy"
GLOVE_VAB_IDX="${GLOVE_DIR}/glove/our_vab_idx.pkl"
GLOVE_VAB_WORDS="${GLOVE_DIR}/glove/our_vab_words.pkl"

if [ -f "$GLOVE_VAB_DATA" ] && [ -f "$GLOVE_VAB_IDX" ] && [ -f "$GLOVE_VAB_WORDS" ]; then
    echo "  ✓ GloVe vocabulary already exists at ${GLOVE_DIR}/glove/"
else
    echo "  GloVe vocabulary not found."
    echo ""
    echo "  The evaluation code does NOT use the raw Stanford GloVe vectors; it needs"
    echo "  the processed 'glove' folder distributed with the HumanML3D repository"
    echo "  (our_vab_data.npy / our_vab_idx.pkl / our_vab_words.pkl):"
    echo "  https://github.com/EricGuo5513/HumanML3D"
    echo ""
    echo "  Place the folder so that the following paths exist:"
    echo "  -----------------------------------------------"
    echo "  mkdir -p ${GLOVE_DIR}/glove"
    echo "  # ${GLOVE_VAB_DATA}"
    echo "  # ${GLOVE_VAB_IDX}"
    echo "  # ${GLOVE_VAB_WORDS}"
    echo "  -----------------------------------------------"
fi

# ==================== Summary ====================
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

if [ -f "$T2M_CHECKPOINT" ]; then
    echo "  ✓ T2M Evaluator (finest.tar): Ready"
else
    echo "  ✗ T2M Evaluator (finest.tar): Not found (required for text-motion metrics)"
fi

if [ -f "$T2M_META_MEAN" ] && [ -f "$T2M_META_STD" ]; then
    echo "  ✓ Evaluator meta stats (Comp_v6_KLD01/meta): Ready"
else
    echo "  ✗ Evaluator meta stats (Comp_v6_KLD01/meta): Not found (metrics silently use training-set stats without them)"
fi

if [ -f "$GLOVE_VAB_DATA" ] && [ -f "$GLOVE_VAB_IDX" ] && [ -f "$GLOVE_VAB_WORDS" ]; then
    echo "  ✓ GloVe vocabulary (our_vab_*): Ready"
else
    echo "  ✗ GloVe vocabulary (our_vab_*): Not found (required for text encoding)"
fi

echo ""
echo "To run evaluation:"
echo "  bash test_stage4.sh"
echo ""
echo "=========================================="
