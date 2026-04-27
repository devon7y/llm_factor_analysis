#!/bin/bash
#SBATCH --account=def-jcaplan
#SBATCH --job-name=embed_mig
#SBATCH --output=/home/devon7y/scratch/devon7y/llm/logs/%x_%j.out
#SBATCH --error=/home/devon7y/scratch/devon7y/llm/logs/%x_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1

set -euo pipefail

source ~/myenv/bin/activate
cd /home/devon7y/scratch/devon7y/llm

export HF_HOME=/home/devon7y/scratch/devon7y/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export XDG_CACHE_HOME="${SLURM_TMPDIR}/xdg-cache"
export HF_MODULES_CACHE="${SLURM_TMPDIR}/hf-modules"
export TORCH_HOME="${SLURM_TMPDIR}/torch-cache"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export PYTHONDONTWRITEBYTECODE=1

MODEL_SRC=/home/devon7y/scratch/devon7y/huggingface/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
MODEL_DST="${SLURM_TMPDIR}/Qwen3-Embedding-8B"

mkdir -p /home/devon7y/scratch/devon7y/llm/logs /home/devon7y/scratch/devon7y/llm/embeddings
mkdir -p "${XDG_CACHE_HOME}" "${HF_MODULES_CACHE}" "${TORCH_HOME}" "${PYTHONPYCACHEPREFIX}"

echo "[$(date)] Running on $(hostname)"
echo "[$(date)] Staging model from ${MODEL_SRC} to ${MODEL_DST}"
rsync -aL --delete "${MODEL_SRC}/" "${MODEL_DST}/"
du -sh "${MODEL_DST}"

echo "[$(date)] Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Generating TAS-20 embeddings ==="
python -u generate_scale_embeddings_single.py \
    --scale-csv scale_items/TAS-20_items.csv \
    --model-name "${MODEL_DST}" \
    --output embeddings/TAS-20_items_8B.npz \
    --normalize-embeddings

echo "=== Generating AQ-50 embeddings ==="
python -u generate_scale_embeddings_single.py \
    --scale-csv scale_items/AQ-50_items.csv \
    --model-name "${MODEL_DST}" \
    --output embeddings/AQ-50_items_8B.npz \
    --normalize-embeddings

echo "=== Done ==="
echo "[$(date)] Finished"
ls -lh embeddings/TAS-20_items_8B.npz embeddings/AQ-50_items_8B.npz
