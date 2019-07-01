#!bin/bash

# Nips
# With n0 = 2
# Estimated initiator     [0.969639, 0.588283; 0.588283, 0.0592866]
# [0.985731, 0.580984; 0.580984, 0.0519559]

RANDOM=110104
for i in {01..25}
do
file="nips_$i.tsv"
echo $file
./krongen/krongen -m:"0.969639, 0.588283; 0.588283, 0.0592866" -i:11 -s:$RANDOM -o:$file
sed -i '1,4d' $file
done

