#!/bin/bash
# Train state estimation model
#
# Usage:
#   bash scripts/train.sh gru 0          # train GRU on GPU 0 (SO(3))
#   bash scripts/train.sh gru 0 so2      # train GRU on GPU 0 (SO(2))
#   bash scripts/train.sh lstm 0,1       # train LSTM on GPU 0 and 1
#   bash scripts/train.sh rnn 2,3,4      # train RNN on GPU 2, 3, 4
#   bash scripts/train.sh gru all        # train GRU on all GPUs
#
# Defaults:
#   - Data: data/estimation/trajectories.pt (or trajectories_so2.pt)
#   - WandB logging enabled
#   - 200 epochs, batch_size=128, lr=1e-3, dropout=0.5
#   - Gradient clipping=1.0, CosineAnnealing LR schedule

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
MODEL_TYPE="${1:-gru}"
GPUS="${2:-0}"
MANIFOLD="${3:-so3}"

# ── Config (edit these as needed) ─────────────────────────────────────────────
EPOCHS=200
BATCH_SIZE=128
LR=1e-3
HIDDEN_SIZE=256
N_PLACE_CELLS=64
DROPOUT=0.5
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SEED=42
WANDB_PROJECT="articulated-estimation"

# Manifold-dependent config
if [ "$MANIFOLD" = "so2" ]; then
    DATA_PATH="data/estimation/trajectories_so2.pt"
    INPUT_SIZE=2
    INIT_POS_SIZE=4
    OUTPUT_DIR="logs/estimation/${MODEL_TYPE}_so2"
else
    DATA_PATH="data/estimation/trajectories.pt"
    INPUT_SIZE=6
    INIT_POS_SIZE=$N_PLACE_CELLS
    OUTPUT_DIR="logs/estimation/${MODEL_TYPE}"
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="python"
fi

# ── GPU handling ──────────────────────────────────────────────────────────────
if [ "$GPUS" = "all" ]; then
    NUM_DEVICES=-1  # Lightning auto-detects all GPUs
    export CUDA_VISIBLE_DEVICES=""  # don't filter, let Lightning see all
else
    export CUDA_VISIBLE_DEVICES="$GPUS"
    NUM_DEVICES=$(echo "$GPUS" | awk -F',' '{print NF}')
fi

if [ "$NUM_DEVICES" -gt 1 ] || [ "$NUM_DEVICES" -eq -1 ]; then
    STRATEGY="ddp"
else
    STRATEGY="auto"
fi

# ── Print config ──────────────────────────────────────────────────────────────
echo "============================================"
echo "  State Estimation Training"
echo "============================================"
echo "Model:      ${MODEL_TYPE^^} (hidden=${HIDDEN_SIZE}, dropout=${DROPOUT})"
echo "Manifold:   ${MANIFOLD} (input=${INPUT_SIZE}, init_pos=${INIT_POS_SIZE})"
echo "Data:       ${DATA_PATH}"
echo "GPUs:       ${GPUS} (devices=${NUM_DEVICES}, strategy=${STRATEGY})"
echo "Training:   ${EPOCHS} epochs, lr=${LR}, batch=${BATCH_SIZE}"
echo "Output:     ${OUTPUT_DIR}"
echo "WandB:      ${WANDB_PROJECT}"
echo "============================================"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
exec "$PYTHON" "$SCRIPT_DIR/train.py" \
    --data_path "$DATA_PATH" \
    --model_type "$MODEL_TYPE" \
    --hidden_size "$HIDDEN_SIZE" \
    --n_place_cells "$N_PLACE_CELLS" \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --grad_clip "$GRAD_CLIP" \
    --epochs "$EPOCHS" \
    --accelerator gpu \
    --devices "$NUM_DEVICES" \
    --strategy "$STRATEGY" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --manifold "$MANIFOLD" \
    --input_size "$INPUT_SIZE" \
    --init_pos_size "$INIT_POS_SIZE" \
    --wandb \
    --project "$WANDB_PROJECT"
