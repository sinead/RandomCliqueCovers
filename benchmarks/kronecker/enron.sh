#!bin/bash

# Nips
# With n0 = 2
# Estimated initiator [0.994746, 0.617899; 0.617899, 0.0322486] 

RANDOM=110104
for i in {01..25}
do
file="enron_$i.tsv"
echo $file
./krongen/krongen -m:"0.994746, 0.617899; 0.617899, 0.0322486" -i:11 -s:$RANDOM -o:$file
sed -i '1,4d' $file
done

