#!/bin/bash -l
#SBATCH --time=0-04:10:00
#SBATCH --job-name=logicood_adversarial
#SBATCH --partition=gpu-stud
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output/job.out.%j
#SBATCH --error=output/job.err.%j

# ==== Load Modules ====
source /opt/spack/main/env.sh
module purge
module load python/3.13.2-qqnwgra
module load py-virtualenv/20.26.5-qkqoooa

# ==== Setup Virtual Environment ====
VENV_DIR="$HOME/mls_venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new virtual environment..."
    virtualenv --system-site-packages -p python3 "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing PyTorch and requirements..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r "$HOME/code/requirements.txt"
else
    echo "Using existing virtual environment..."
    source "$VENV_DIR/bin/activate"
fi

# ==== Debug: Check Virtual Environment ====
echo "=== Virtual Environment Debug ==="
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ No virtual environment active!"
else
    echo "✅ Virtual environment active at: $VIRTUAL_ENV"
fi
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
echo "================================="

# ==== Confirm Installation ====
echo "=== Python Version ==="
python --version
echo "=== Torch Check ==="
python -c "import torch; print('Torch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"

# ==== Run script ====
srun python "$HOME/code/create_segments.py"
