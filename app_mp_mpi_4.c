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

void *read_points(char *filepath, int n, double* x, double* y);
double calc_distance(double p1_x, double p1_y, double p2_x, double p2_y);
void write_new_centroids(double x[], double y[], int n_centroids, char *output);

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
        - Output file.
*/
int main(int argc, char *argv[])
{
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    //MPI Initiliazation
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int done = 0;

    //Parameters
    int n_points = atoi(argv[2]);
    int n_centroids = atoi(argv[4]);
    int dist_algo = atoi(argv[5]);
    double error = atof(argv[6]);
    char *output_file = argv[7];
    int n_threads = argv[8];

    //Creating data structures

    //Partial array of points
    double *point_x = malloc(sizeof(double) * n_points / world_size);
    double *point_y = malloc(sizeof(double) * n_points / world_size);

    //Total array of points
    double *p_x = malloc(sizeof(double) * n_points);
    double *p_y = malloc(sizeof(double) * n_points);

    //Total array of centroids
    double *centroid_x = malloc(sizeof(double) * n_centroids);
    double *centroid_y = malloc(sizeof(double) * n_centroids);

    //Reading data
    if (world_rank == MASTER)
    {
        read_points(argv[1], n_points, p_x, p_y);
        read_points(argv[3], n_centroids, centroid_x, centroid_y);
    }

    //Scattering points
    //Splits the points across all nodes
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

    //Support data structures

    //Total array of the new centroids
    double *new_centroid_x = malloc(n_centroids * sizeof(double));
    double *new_centroid_y = malloc(n_centroids * sizeof(double));
    //Counts the number of points associated to the new centroid
    int *new_centroids_n_points = malloc(n_centroids * sizeof(int));
    //For every points saves which centroid is the nearest
    int *assign_centroid = malloc(n_points* sizeof(int));
    //Partial array of new centroids
    double *local_new_centroid_x = malloc(n_centroids/world_size * sizeof(double));
    double *local_new_centroid_y = malloc(n_centroids/world_size * sizeof(double));
    //Counts the number of points associated to the new local centroid
    int *local_new_centroids_n_points = malloc(n_centroids/world_size * sizeof(int));
    //
    double *global_new_centroid_x = malloc(n_centroids * sizeof(double));
    double *global_new_centroid_y = malloc(n_centroids * sizeof(double));
    //
    int *global_new_centroids_n_points = malloc(n_centroids * sizeof(int));
    //Indicates for each node how many centroid it will receive
    int *recvcounts = malloc(world_size * sizeof(int));
    double curr_error;

    //Initialization of the recv buffer
    //It stands for how many centroids each node receives
    for(int i = 0; i < world_size; i++) {
        recvcounts[i] = n_centroids / world_size;
    }

    //Start computation
    int iteration = 0;


    //Broadcasting centroids
    //All nodes will know the centroids
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

    do
    {

        //Resetting support data structures
        curr_error = 0;

        //Initialize the Total new centroids array to 0
        #pragma omp parallel for
        for (int i = 0; i < n_centroids; i++)
        {
            new_centroid_x[i] = 0;
            new_centroid_y[i] = 0;
            new_centroids_n_points[i] = 0;
        }

        //Initialize the Local new centroids array to 0
        #pragma omp parallel for
        for (int i = 0; i < n_centroids / world_size; i++)
        {
            local_new_centroid_x[i] = 0;
            local_new_centroid_y[i] = 0;
            local_new_centroids_n_points[i] = 0;
        }

        //For each point, look for the closest centroid and
        //assing the point to the centroid.
        #pragma omp parallel for reduction(+: new_centroid_x[:n_centroids],new_centroid_y[:n_centroids], new_centroids_n_points[:n_centroids])
        for (int i = 0; i < n_points / world_size; i++)
        {
            double min_distance = INVALID;
            int closest_centroid = INVALID;

            for (int j = 0; j < n_centroids; j++)
            {

                double distance = calc_distance(point_x[i], point_y[i], centroid_x[j], centroid_y[j]);
                if (distance < min_distance || min_distance == INVALID)
                {
                    min_distance = distance;
                    closest_centroid = j;
                }

            }
            //Increment the partial record. 
            //It will be averaged.
            //#pragma omp critical
            //{
                //Sum the coordinates of the points asssociated to the closest_centroid 
                //(the new centroid will be calculated later by averaging this two arrays with the number of points associated to the closest_centroid)
                new_centroid_x[closest_centroid] += point_x[i];
                new_centroid_y[closest_centroid] += point_y[i];
                //Count the points associated to the closest centroid
                new_centroids_n_points[closest_centroid]++;
            //  }
            
        }
        //Reduce the partial computation.
        //Scatter all new centroids across all nodes.
        //Reduce all the new_centroid arrays into one and then scatter again across all nodes

        //At the first iteration each node has a partition of the points and the full set of the centroids.
        //In the other iterations each node will have a partition of the points and a partiton of the centroids.
        MPI_Reduce_scatter(
            new_centroid_x,
            local_new_centroid_x,
            recvcounts,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);

        MPI_Reduce_scatter(
            new_centroid_y,
            local_new_centroid_y,
            recvcounts,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD);
        
        MPI_Reduce_scatter(
            new_centroids_n_points,
            local_new_centroids_n_points,
            recvcounts,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);
        
        //offset indicates which portion of the new_centroid arrays is assigned to the current node.
        //i.e the centroids are 20
        //Node 1: new_centroids_*[0:9]
        //Node 2: new_centroids_*[10:19]

        int offset = world_rank * (n_centroids / world_size);
        
        #pragma omp parallel for reduction(+: curr_error)
        //Iterate over the partition of the new_centroids
        for (int i = 0; i < n_centroids/world_size; i++)
        {
            if (local_new_centroids_n_points[i] != 0)
            {
                //Averaging the Local new centroids
                local_new_centroid_x[i] = local_new_centroid_x[i] / local_new_centroids_n_points[i];
                local_new_centroid_y[i] = local_new_centroid_y[i] / local_new_centroids_n_points[i];
                //Calculate the local distance from the old one
                //centroid_* : Total array of centroids
                //local_new_centroids_* : Local array of the new centroids
                curr_error += calc_distance(centroid_x[offset + i], centroid_y[offset + i], local_new_centroid_x[i], local_new_centroid_y[i]);
            } else {
                //If no points are associated to the centroid, it will write the old value of the centroid
                local_new_centroid_x[i] = centroid_x[offset+i];
                local_new_centroid_y[i] = centroid_y[offset+i];
            }
        }

        //Sum all curr_error
        //Broadcast the result to all node.
        MPI_Allreduce(
            &curr_error,
            &curr_error,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );
        //Gather all the centroids and send the total array to all nodes
        MPI_Allgather(
            local_new_centroid_x,
            n_centroids / world_size,
            MPI_DOUBLE,
            centroid_x,
            n_centroids / world_size,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        MPI_Allgather(
            local_new_centroid_y,
            n_centroids / world_size,
            MPI_DOUBLE,
            centroid_y,
            n_centroids / world_size,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        //Break if the computation reached enough precision
        if (curr_error < error)
        {
            break;
        }
        iteration++;
	if(world_rank == MASTER)
        printf("Current error : %lf -- Curr iteration : %d\n", curr_error, iteration);

    } while (done == 0);

    gettimeofday(&end_time, NULL);
    MPI_Finalize();

    printf("Total number of iterations: %d\n", iteration);
    printf("Time elapsed is %lu\n", end_time.tv_sec - start_time.tv_sec);

    //Write the centroids to file
    if(world_rank == MASTER){
	    //write_new_centroids(centroid_x, centroid_y, n_centroids, output_file);
    }

    return 0;
}

void write_new_centroids(double x[], double y[], int n_centroids, char *output) {
	int i;
	FILE *fp;
	fp = fopen(output, "w");

	if(fp == NULL) {
		perror("Error while opening the file\n");
		exit(EXIT_FAILURE);
	}

	for(i = 0; i < n_centroids; i++) {
		printf("x %lf y %lf iteration i %d\n", x[i], y[i], i);
		fprintf(fp, "%lf %lf\n", x[i],y[i]);
	}
	fclose(fp);
}

void *read_points(char *filepath, int n, double* x, double* y)
{
    int i;
    FILE *fp;
    fp = fopen(filepath, "r"); // read mode

    if (fp == NULL)
    {
        perror("Error while opening the file.\n");
        exit(EXIT_FAILURE);
    }

    for ( i = 0; i < n; i++)
    {
        char string[BUFFER_SIZE];
        fgets(string, BUFFER_SIZE, fp);
        sscanf(string, "%lf %lf", &(x[i]), &(y[i]));
        //print_point(&points[i]);
    }

    fclose(fp);
}

inline double calc_distance(double p1_x, double p1_y, double p2_x, double p2_y)
{
	double delta_x = p1_x - p2_x;
	double delta_y = p1_y - p2_y;
    return sqrt((delta_x) * (delta_x) + (delta_y) * (delta_y));
}

void print_point(point *p)
{
    printf("Printing point\n");
    printf(" -- X coordinate : %lf\n", p->x);
    printf(" -- Y coordinate : %lf\n", p->y);
}
