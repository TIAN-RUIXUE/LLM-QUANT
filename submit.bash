#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00          # must be <= your QOS MaxWall; lower it if still pending
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=finllm
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Initialise conda
eval "$(conda shell.bash hook)"
conda activate quant

# Ollama: local-only, and a per-job port so we don't clash with other users on the node
PORT=11434
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_CONTEXT_LENGTH=8192   # default window is too small for definitions + news

ollama serve > "$HOME/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT   # always stop ollama when the job ends

# Wait until the server actually responds (up to 2 min) instead of a fixed sleep 5
for i in $(seq 1 60); do
    if curl -s "http://$OLLAMA_HOST/api/tags" > /dev/null; then
        echo "Ollama ready on $OLLAMA_HOST"
        break
    fi
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "Ollama failed to start, see ~/ollama_${SLURM_JOB_ID}.log" >&2
        exit 1
    fi
    sleep 2
done

nvidia-smi

# Run your code
python main.py