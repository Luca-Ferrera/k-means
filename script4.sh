#!/bin/bash
cd ../docker.openmpi
docker cp ../k-means/app_mp_mpi_4.c dockeropenmpi_mpi_head_1:home/
docker cp ../k-means/points.in dockeropenmpi_mpi_head_1:home/
docker cp ../k-means/centroids.in dockeropenmpi_mpi_head_1:home/
docker exec -i -t dockeropenmpi_mpi_head_1 bash -c "cd /home && mpicc -o app_mp_mpi4 app_mp_mpi_4.c -lm && time mpirun -np $1 -x OMP_NUM_THREADS=$2 --allow-run-as-root app_mp_mpi4 points.in $3 centroids.in $4 1 1 new_centroids.in"