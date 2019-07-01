#!bin/bash

# Arxiv
# With n0 = 2
# initiator="0.9999, 0.195826; 0.244085, 0.615717"
# With no0 = 4
initiator="0.9999, 0.408522, 0.0193124, 0.0001; 0.415092, 0.599702, 0.22658, 0.0179405; 0.0110087, 0.325344, 0.256641, 0.0287054; 0.00505781, 0.0324692, 0.0529437, 0.403938"
RANDOM=110104

for i in {01..25}
do
file="arxiv_$i.tsv"
echo $file
./krongen/krongen -m:"$initiator" -i:7 -s:$RANDOM -o:$file
sed -i '1,4d' $file
done

