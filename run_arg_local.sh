start=$(date +"%s")
echo Start time: `date +"%T"`

mkdir -p temp/news 
mkdir -p temp/arguments

python code/arg-class.py
echo "End process"


end=$(date +"%s")
echo End time: `date +"%T"`

rm -r temp

diff=$(expr $end - $start)
echo Runtime: $diff seconds