#!/bin/bash
#Execute the script inside k-means folder!

#Execute the program
./script1.sh 5 $1 $2

#find where the docker image store new_centroids.in
find / -name new_centroids.in > partials
sed 1q partials > final

path=`cat final`

#remove files
rm partials
rm final

#move it to k-means folder
mv ${path} ../k-means/

#plotting
python3 plot.py $1 $2
