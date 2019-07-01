#!bin/bash

# Nips
# With n0 = 2
estimator="0.700513, 0.427455; 0.417285, 0.9999"

RANDOM=110104
for i in {01..25}
do
file="../../../sparsedense3_paperversion/facebook/krongen_runs/facebook_$i.tsv"
echo $fileLS
./krongen/krongen -m:"$estimator" -i:12 -s:$RANDOM -o:$file
sed -i '1,4d' $file
done

