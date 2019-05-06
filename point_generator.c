#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double randfrom(double min, double max);

/*
    Arguments:
        - Filename.
        - Number of points.
        - Min X coordinate.
        - Max X coordinate.
        - Min Y coordinate.
        - Max Y coordinate.
*/
int main(int argc, char *argv[])
{
    char *filename = argv[1];
    int n_points = atoi(argv[2]);
    double x_min = atof(argv[3]);
    double x_max = atof(argv[4]);
    double y_min = atof(argv[5]);
    double y_max = atof(argv[6]);

    FILE *f = fopen(filename, "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    srand(time(NULL));

    for (int i = 0; i < n_points; i++)
    {
        double x = randfrom(x_min, x_max);
        double y = randfrom(y_min, y_max);
        fprintf(f, "%lf %lf\n", x, y);
    }

    fclose(f);
}

double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}