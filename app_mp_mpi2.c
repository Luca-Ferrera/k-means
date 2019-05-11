#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define BUFFER_SIZE 200
#define INVALID -1
#define MASTER 0

typedef struct
{
    double x;
    double y;
} point;

point *read_points(char *filepath, int n);
double calc_distance(double p1_x, double p1_y, double p2_x, double p2_y, int algo);

void print_point(point *p);

/*
    Arguments:
        - Points file path.
        - Number of points.
        - Centroids file path.
        - Number of centroids.
        - Type of distance:
            - 0, Manhattan
            - 1, Euclidean
            - 2, Euclidean no SQRT
        - Error.
*/
int main(int argc, char *argv[])
{
    int i,j;
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    //MPI Initiliazation
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    printf("world_size: %d\n", world_size);

    int done = 0;

    //Parameters
    int n_points = atoi(argv[2]);
    int n_centroids = atoi(argv[4]);
    int dist_algo = atoi(argv[5]);
    double error = atof(argv[6]);

    //Creating data structures
    double *point_x = malloc(sizeof(double) * n_points / world_size);
    double *point_y = malloc(sizeof(double) * n_points / world_size);
    double *p_x = malloc(sizeof(double) * n_points);
    double *p_y = malloc(sizeof(double) * n_points);
    double *centroid_x = malloc(sizeof(double) * n_centroids);
    double *centroid_y = malloc(sizeof(double) * n_centroids);
    
    //Support data structures
    double *new_centroid_x = malloc(n_centroids * sizeof(double));
    double *new_centroid_y = malloc(n_centroids * sizeof(double));
    int *new_centroids_n_points = malloc(n_centroids * sizeof(int));
    int *global_new_centroids_n_points = malloc(n_centroids * sizeof(int));
    double *global_new_centroid_x = malloc(n_centroids * sizeof(double));
    double *global_new_centroid_y = malloc(n_centroids * sizeof(double));
    double global_curr_error;

    //create data structures for divide the centroids calculation load 
    double *local_new_centroids_x = malloc(n_centroids/world_size * sizeof(double));
    double *local_new_centroids_y = malloc(n_centroids/world_size * sizeof(double));
    double *local_centroids_x = malloc(n_centroids/world_size * sizeof(double));
    double *local_centroids_y = malloc(n_centroids/world_size * sizeof(double));
    int *local_centroids_n_points = malloc(n_centroids/world_size *sizeof(int));
    double local_curr_error;
    //migliorare read_points come array di variabili primitive invece di usare le struct
    point *points;
    point *centroids;

    //Reading data
    if (world_rank == MASTER)
    {
        points = read_points(argv[1], n_points);
        centroids = read_points(argv[3], n_centroids);
        //openMP per splittare il for DONE
        #pragma omp parallel for 
        for (i = 0; i < n_points; i++)
        {
            p_x[i] = points[i].x;
            p_y[i] = points[i].y;
        }
        #pragma omp parallel for
        for (i = 0; i < n_centroids; i++)
        {
            centroid_x[i] = centroids[i].x;
            centroid_y[i] = centroids[i].y;
        }
    }

    //Scattering points
    MPI_Scatter(
        p_x,
        n_points / world_size,
        MPI_DOUBLE,
        point_x,
        n_points / world_size,
        MPI_DOUBLE,
        MASTER,
        MPI_COMM_WORLD);
    MPI_Scatter(
        p_y,
        n_points / world_size,
        MPI_DOUBLE,
        point_y,
        n_points / world_size,
        MPI_DOUBLE,
        MASTER,
        MPI_COMM_WORLD);

    //Start computation
    int iteration = 0;

    do
    {

        //Broadcasting centroids
        MPI_Bcast(
            centroid_x,
            n_centroids,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD);
        MPI_Bcast(
            centroid_y,
            n_centroids,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD);

        //Resetting support data structures
        global_curr_error = 0;
        local_curr_error = 0;
        #pragma omp parallel for
        for (i = 0; i < n_centroids; i++)
        {
            new_centroid_x[i] = 0;
            new_centroid_y[i] = 0;
            new_centroids_n_points[i] = 0;
        }

        //For each point, look for the closest centroid and
        //assing the point to the centroid.
        #pragma omp parallel for
        for (i = 0; i < n_points / world_size; i++)
        {

            double min_distance = INVALID;
            int closest_centroid = INVALID;

            for (j = 0; j < n_centroids; j++)
            {

                double distance = calc_distance(point_x[i], point_y[i], centroid_x[j], centroid_y[j], dist_algo);

                if (distance < min_distance || min_distance == INVALID)
                {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }

            #pragma omp critical
            {
                new_centroid_x[closest_centroid] += point_x[i];
                new_centroid_y[closest_centroid] += point_y[i];
                new_centroids_n_points[closest_centroid]++;
            }
        }
        
        MPI_Reduce(
            new_centroid_x,
            global_new_centroid_x,
            n_centroids,
            MPI_DOUBLE,
            MPI_SUM,
            MASTER,
            MPI_COMM_WORLD);

        MPI_Reduce(
            new_centroid_y,
            global_new_centroid_y,
            n_centroids,
            MPI_DOUBLE,
            MPI_SUM,
            MASTER,
            MPI_COMM_WORLD);
        
        MPI_Reduce(
            new_centroids_n_points,
            global_new_centroids_n_points,
            n_centroids,
            MPI_INT,
            MPI_SUM,
            MASTER,
            MPI_COMM_WORLD);
        


        //divide centroids to peers
        if(world_rank == MASTER){
            for(i = 0; i < n_centroids; i++){
                //printf("Iteration: %d GCX: %f GCY: %f\n", iteration, centroid_x[i], centroid_y[i]);
                //printf("Iteration: %d GNCX: %f GNCY: %f\n",iteration, global_new_centroid_x[i], global_new_centroid_y[i]);
                
            }
        }

        MPI_Scatter(
            centroid_x,
            n_centroids/world_size,
            MPI_DOUBLE,
            local_centroids_x,
            n_centroids/world_size,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD
        );
        MPI_Scatter(
            centroid_y,
            n_centroids/world_size,
            MPI_DOUBLE,
            local_centroids_y,
            n_centroids/world_size,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD
        );
        MPI_Scatter(
            global_new_centroid_x,
            n_centroids/world_size,
            MPI_DOUBLE,
            local_new_centroids_x,
            n_centroids/world_size,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD
        );
        MPI_Scatter(
            global_new_centroid_y,
            n_centroids/world_size,
            MPI_DOUBLE,
            local_new_centroids_y,
            n_centroids/world_size,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD
        );
        MPI_Scatter(
            new_centroids_n_points,
            n_centroids/world_size,
            MPI_INT,
            local_centroids_n_points,
            n_centroids/world_size,
            MPI_INT,
            MASTER,
            MPI_COMM_WORLD
        );
            
        //***maybe useless***
        #pragma omp parallel for reduction(+: local_curr_error)
        for(i = 0; i < n_centroids/world_size; i++){
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (local_centroids_n_points[i] != 0){
                
                local_new_centroids_x[i] = local_new_centroids_x[i] / local_centroids_n_points[i];
                local_new_centroids_y[i] = local_new_centroids_y[i] / local_centroids_n_points[i];
                local_curr_error += calc_distance(local_centroids_x[i], local_centroids_y[i], local_new_centroids_x[i], local_new_centroids_y[i], dist_algo);
                local_centroids_x[i] = local_new_centroids_x[i];
                local_centroids_y[i] = local_new_centroids_y[i];
            }
        }
         
        MPI_Gather(
            local_centroids_x,
            n_centroids/world_size,
            MPI_DOUBLE,
            centroid_x,
            n_centroids/world_size,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD
        );
        MPI_Gather(
            local_centroids_y,
            n_centroids/world_size,
            MPI_DOUBLE,
            centroid_y,
            n_centroids/world_size,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD
        );
        MPI_Gather(
            local_new_centroids_x,
            n_centroids/world_size,
            MPI_DOUBLE,
            global_new_centroid_x,
            n_centroids/world_size,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD
        );
        MPI_Gather(
            local_new_centroids_y,
            n_centroids/world_size,
            MPI_DOUBLE,
            global_new_centroid_y,
            n_centroids/world_size,
            MPI_DOUBLE,
            MASTER,
            MPI_COMM_WORLD
        );
        MPI_Gather(
            local_centroids_n_points,
            n_centroids/world_size,
            MPI_INT,
            new_centroids_n_points,
            n_centroids/world_size,
            MPI_INT,
            MASTER,
            MPI_COMM_WORLD
        );
        MPI_Reduce(
            &local_curr_error,
            &global_curr_error,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MASTER,
            MPI_COMM_WORLD
        );
                
        if(world_rank == MASTER){
            printf("Current global error : %lf -- Curr iteration : %d\n", global_curr_error, iteration);
            iteration++;
            if (global_curr_error < error)
            {
                done = 1;
            }

        }
        
        MPI_Bcast(
            &done,
            1,
            MPI_INT,
            MASTER,
            MPI_COMM_WORLD);

    } while (done == 0);
    

    gettimeofday(&end_time, NULL);

    MPI_Finalize();

    printf("Time elapsed is %lu\n", end_time.tv_sec - start_time.tv_sec);

    return 0;
}

point *read_points(char *filepath, int n)
{
    int i;
    FILE *fp;
    fp = fopen(filepath, "r"); // read mode

    if (fp == NULL)
    {
        perror("Error while opening the file.\n");
        exit(EXIT_FAILURE);
    }

    //Create the point array
    point *points = malloc(n * sizeof(point));

    for (i = 0; i < n; i++)
    {
        char string[BUFFER_SIZE];
        fgets(string, BUFFER_SIZE, fp);
        sscanf(string, "%lf %lf", &(points[i].x), &(points[i].y));
        //print_point(&points[i]);
    }

    fclose(fp);
    return points;
}

double calc_distance(double p1_x, double p1_y, double p2_x, double p2_y, int algo)
{

    switch (algo)
    {
    case 0:
        return fabs(p1_x + p1_y - p2_x - p2_y);
        break;

    case 1:
        return sqrt(pow(p1_x - p2_x, 2) + pow(p1_y - p2_y, 2));
        break;

    case 2:
        return pow(p1_x - p2_x, 2) + pow(p1_y - p2_y, 2);
        break;

    default:
        return 0;
        break;
    }
}

void print_point(point *p)
{
    printf("Printing point\n");
    printf(" -- X coordinate : %lf\n", p->x);
    printf(" -- Y coordinate : %lf\n", p->y);
}