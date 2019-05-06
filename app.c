#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>

#define BUFFER_SIZE 200
#define INVALID -1

typedef struct
{
    double x;
    double y;
} point;

point *read_points(char *filepath, int n);
double calc_distance(point p1, point p2, int algo);

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

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    //Parameters
    int n_points = atoi(argv[2]);
    int n_centroids = atoi(argv[4]);
    int dist_algo = atoi(argv[5]);
    double error = atof(argv[6]);

    //Reading data
    point *points = read_points(argv[1], n_points);
    point *centroids = read_points(argv[3], n_centroids);

    //Support data structures
    point *new_centroids = malloc(n_centroids * sizeof(point));
    int *new_centroids_n_points = malloc(n_centroids * sizeof(int));
    double curr_error;

    //Start computation
    int iteration = 0;

    do
    {
        //Resetting support data structures
        curr_error = 0;
        for (int i = 0; i < n_centroids; i++)
        {

            new_centroids[i].x = 0;
            new_centroids[i].y = 0;
            new_centroids_n_points[i] = 0;
        }

        //For each point, look for the closest centroid and
        //assing the point to the centroid.
        for (int i = 0; i < n_points; i++)
        {

            double min_distance = INVALID;
            int closest_centroid = INVALID;

            for (int j = 0; j < n_centroids; j++)
            {

                double distance = calc_distance(points[i], centroids[j], dist_algo);

                    if (distance < min_distance || min_distance == INVALID)
                    {
                        min_distance = distance;
                        closest_centroid = j;
                    }
            }

                new_centroids[closest_centroid].x += points[i].x;
                new_centroids[closest_centroid].y += points[i].y;
                new_centroids_n_points[closest_centroid]++;
        }

        //Calculate the new centroids and the error
        for (int i = 0; i < n_centroids; i++)
        {

            if (new_centroids_n_points[i] != 0)
            {
                new_centroids[i].x = new_centroids[i].x / new_centroids_n_points[i];
                new_centroids[i].y = new_centroids[i].y / new_centroids_n_points[i];
                //print_point(&new_centroids[i]);
                curr_error += calc_distance(centroids[i], new_centroids[i], dist_algo);
                centroids[i].x = new_centroids[i].x;
                centroids[i].y = new_centroids[i].y;
            }
        }

        iteration++;
        printf("Current error : %lf -- Curr iteration : %d\n", curr_error, iteration);

    } while (curr_error > error);

    gettimeofday(&end_time, NULL);

    printf("Time elapsed is %lu\n", end_time.tv_sec - start_time.tv_sec);

    return 0;
}

point *read_points(char *filepath, int n)
{

    FILE *fp;
    fp = fopen(filepath, "r"); // read mode

    if (fp == NULL)
    {
        perror("Error while opening the file.\n");
        exit(EXIT_FAILURE);
    }

    //Create the point array
    point *points = malloc(n * sizeof(point));

    for (int i = 0; i < n; i++)
    {
        char string[BUFFER_SIZE];
        fgets(string, BUFFER_SIZE, fp);
        sscanf(string, "%lf %lf", &(points[i].x), &(points[i].y));
        //print_point(&points[i]);
    }

    fclose(fp);
    return points;
}

double calc_distance(point p1, point p2, int algo)
{

    switch (algo)
    {
    case 0:
        return fabs(p1.x + p1.y - p2.x - p2.y);
        break;

    case 1:
        return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
        break;

    case 2:
        return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
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