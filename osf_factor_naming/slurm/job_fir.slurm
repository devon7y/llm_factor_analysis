#!/bin/bash
#SBATCH --account=def-jcaplan
#SBATCH --job-name=fn_paper
#SBATCH --output=/home/devon7y/scratch/devon7y/factor_naming_paper/logs/%x_%j.out
#SBATCH --error=/home/devon7y/scratch/devon7y/factor_naming_paper/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=125G
#SBATCH --gpus=h100:1

# The exact job that produced the paper's numbers (Fir, Alliance Canada).
# Everything analysis-relevant lives in the R scripts; this file only
# provisions the environment.

set -euo pipefail
module load StdEnv/2023 gcc r/4.4.0 python/3.11

PROJ=/home/devon7y/scratch/devon7y/factor_naming_paper

export R_LIBS=$PROJ/R_libs
export RETICULATE_PYTHON=$PROJ/venv/bin/python
export R_USER_CACHE_DIR=$PROJ/rcache        # semanticfa embedding + pool cache
export HF_HOME=$PROJ/hf                     # model weights cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

# Fail fast on a broken GPU allocation rather than embedding on CPU.
$RETICULATE_PYTHON - <<'PY'
import torch, sys
ok = torch.cuda.is_available()
print("CUDA available:", ok, "| device:", torch.cuda.get_device_name(0) if ok else "none")
sys.exit(0 if ok else 2)
PY

cd $PROJ/osf_factor_naming
Rscript run_all.R
