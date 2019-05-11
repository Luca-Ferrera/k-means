#!/bin/bash
cd ../docker.openmpi
docker cp ../k-means/app_mp_mpi2.c dockeropenmpi_mpi_head_1:home/
docker cp ../k-means/points.in dockeropenmpi_mpi_head_1:home/
docker cp ../k-means/centroids.in dockeropenmpi_mpi_head_1:home/
docker exec -i -t dockeropenmpi_mpi_head_1 bash -c "cd /home && mpicc -o app_mp_mpi2 app_mp_mpi2.c -lm && time mpirun -np $1 --allow-run-as-root app_mp_mpi2 points.in $2 centroids.in $3 1 1"