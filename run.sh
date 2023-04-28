#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=Preprocessing
#SBATCH --output=preprocessing.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH -e errfile
#SBATCH --partition=thin

now=$(date + "%T")
echo "Start time: $now"
cd $HOME/fake_news

# Activate
source "env/bin/activate"
# pip install -r $HOME/thesis/requirements.txt

# Run your code
srun echo "Start process"
srun python code/train.py
srun echo "End process"

end=$(date + "%T")
echo "Start time: $end"

deactivate
=======

python code/train.py
>>>>>>> d92ce811516b2f596f7267dac154d827c9a1856c
