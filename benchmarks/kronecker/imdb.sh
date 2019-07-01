#!bin/bash

# Nips
# With n0 = 2
# Estimated initiator [0.9999, 0.151284; 0.150061, 0.740534] 

RANDOM=110104
for i in {01..25}
do
file="imdb_$i.tsv"
echo $file
./krongen/krongen -m:"0.9999, 0.151284; 0.150061, 0.740534" -i:12 -s:$RANDOM -o:$file
sed -i '1,4d' $file
done

