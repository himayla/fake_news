start=$(date +"%s")
echo Start time: `date +"%T"`

# Activate env and install requirements
source "env/bin/activate"
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
python -m pip install -r requirements.txt

# Run your code
echo "Start process"
python code/arg-class.py
echo "End process"

end=$(date +"%s")
echo End time: `date +"%T"`

diff=$(expr $end - $start)
echo Runtime: $diff seconds

deactivate