#!/bin/bash
#SBATCH --account=def-jcaplan
#SBATCH --job-name=embed_tas_aq
#SBATCH --output=/scratch/devon7y/llm_factor_analysis/logs/%x_%j.out
#SBATCH --error=/scratch/devon7y/llm_factor_analysis/logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1

set -euo pipefail

module load python/3.11
source ~/myenv/bin/activate

cd /scratch/devon7y/llm_factor_analysis

export HF_HOME=/scratch/devon7y/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export XDG_CACHE_HOME="${SLURM_TMPDIR}/xdg-cache"
export HF_MODULES_CACHE="${SLURM_TMPDIR}/hf-modules"
export TORCH_HOME="${SLURM_TMPDIR}/torch-cache"
mkdir -p "${XDG_CACHE_HOME}" "${HF_MODULES_CACHE}" "${TORCH_HOME}"

echo "[$(date)] Running on $(hostname)"
echo "[$(date)] Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Generating TAS-20 embeddings ==="
python -u generate_scale_embeddings_single.py \
    --scale-csv scale_items/TAS-20_items.csv \
    --model-name Qwen/Qwen3-Embedding-8B \
    --output embeddings/TAS-20_items_8B.npz \
    --normalize-embeddings

echo "=== Generating AQ-50 embeddings ==="
python -u generate_scale_embeddings_single.py \
    --scale-csv scale_items/AQ-50_items.csv \
    --model-name Qwen/Qwen3-Embedding-8B \
    --output embeddings/AQ-50_items_8B.npz \
    --normalize-embeddings

echo "=== Done ==="
echo "[$(date)] Finished"
ls -lh embeddings/
