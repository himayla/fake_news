#!/bin/bash
#SBATCH --job-name=Training_Kaggle
#SBATCH --output=Training_Kaggle.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=thin
#SBATCH --time=48:00:00
#SBATCH -e terminal_Kaggle


start=$(date +"%s")
echo Start time: `date +"%T"`

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
source "env/bin/activate"
#python -m pip install -r requirements.txt

# Run your code
srun echo "Start process"
srun python code/train.py
srun echo "End process"

end=$(date +"%s")
echo End time: `date +"%T"`

diff=$(expr $end - $start)
echo Runtime: $diff seconds

deactivate
