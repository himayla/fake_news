#!/bin/bash
#SBATCH --job-name=Preprocessing
#SBATCH --output=preprocessing.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH -e errfile
#SBATCH --partition=thin

start=$(date +"%s")
echo Start time: `date +"%T"`

# Activate
source "env/bin/activate"
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
python -m pip install -r requirements.txt

# Run your code
srun echo "Start process"
srun python code/train.py
srun echo "End process"

end=$(date +"%s")
echo End time: `date +"%T"`

diff=$(expr $end - $start)
echo Runtime: $diff seconds

deactivate
