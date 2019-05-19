import sys
import matplotlib.pyplot as plt

#open the files
f = open("points.in", "r")
f1 = open("new_centroids.in", "r")
f2 = open("centroids.in", "r")

#read the number of points and centroids to read
n_points_str = sys.argv[1]
new_centroids_str = sys.argv[2]

#convert it to int
n_points = int(n_points_str,10)
new_centroids = int(new_centroids_str, 10)

print("points " + n_points_str)
print("centroids " + new_centroids_str)

#read the points
for p in range(1, n_points):
    line = f.readline()
    point = line.split(" ")
    print("point " + point[0] + " " + point[1])
    plt.plot(float(point[0]), float(point[1]) , 'bs')

new_centroids_x = []
new_centroids_y = []
old_centroids_x = []
old_centroids_y = []

#read the old and the new centroids, convert to float and split them in x and 7 coordinates
for c in range(1, new_centroids+1):
    
    line1 = f1.readline()
    line2 = f2.readline()
    new_centroid = line1.split(" ")
    old_centroid = line2.split(" ")
    
    new_centroids_x.append(float(new_centroid[0]))
    new_centroids_y.append(float(new_centroid[1]))

    old_centroids_x.append(float(old_centroid[0]))
    old_centroids_y.append(float(old_centroid[1]))

    #Plot the line between old and new centroids
    plt.plot([float(new_centroid[0]), float(old_centroid[0])], [float(new_centroid[1]), float(old_centroid[1])] , linewidth=2)


#plot the old and new centroids
plt.scatter(new_centroids_x, new_centroids_y, marker='o', color='red')
plt.scatter(old_centroids_x, old_centroids_y, marker='x', color='green')

#Range of the plot
plt.axis([0,100, 0, 100])

#show it
plt.show()
