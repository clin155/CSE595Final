#!/bin/bash
#SBATCH --account=cse595s001f24_class
#SBATCH --partition=spgpu
#SBATCH --time=00-08:00:00
#SBATCH --gpus=1
#SBATCH --output=out
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64GB

module load python/3.10.4
module load cuda/12.1

source ~/env/bin/activate
# if diff -q req.txt g.txt > /dev/null; then
#     echo "Not installing packages already match"
# else
# pip install torch==2.4.1 numpy
# pip install bitsandbytes accelerate nltk datasets Pillow trl transformers peft
# pip install scikit-learn

export WANDB_API_KEY="394342629a352f9dad9ce12da4f510caaf3bb5f4"
export HUGGINGFACE_HUB_TOKEN="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"
nvcc --version
python ~/test.py
python ~/finetune_llava.py --force_reload_dataset=True
