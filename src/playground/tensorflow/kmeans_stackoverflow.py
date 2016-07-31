#!/usr/bin/python3
# from http://stackoverflow.com/questions/33621643/how-would-i-implement-k-means-with-tensorflow
# by dga http://stackoverflow.com/users/5545260/dga

#I should probably explain it a little better. It takes the N points and makes K copies of them. It takes the K current centroids and makes N copies of them.
#It then subtracts these two big tensors to get the N*K distances from each point to each centroid. It computes the sum of squared distances from those, and uses 'argmin'
#to find the best one for each point.
#Then it uses dynamic_partition to group the points into K different tensors based upon their cluster assignment, finds the means within each of those clusters, and sets the centroids based upon that. 

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

N=1000
K=4
MAX_ITERS = 100

start = time.time()

points = tf.Variable(tf.random_uniform([N,2]))
cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))

# Silly initialization:  Use the first two points as the starting                
# centroids.  In the real world, do this better.                                 
centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [K,2]))

# Replicate to N copies of each centroid and K copies of each                    
# point, then subtract and compute the sum of squared distances.                 
rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, 2])
rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                            reduction_indices=2)

# Use argmin to select the lowest-distance point                                 
best_centroids = tf.argmin(sum_squares, 1)
did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,
                                                    cluster_assignments))

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count

means = bucket_mean(points, best_centroids, K)

# Do not write to the assigned clusters variable until after                     
# computing whether the assignments have changed - hence with_dependencies
with tf.control_dependencies([did_assignments_change]):
    do_updates = tf.group(
        centroids.assign(means),
        cluster_assignments.assign(best_centroids))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

changed = True
iters = 0

while changed and iters < MAX_ITERS:
    iters += 1
    [changed, _] = sess.run([did_assignments_change, do_updates])

[centers, assignments] = sess.run([centroids, cluster_assignments])
end = time.time()
print ("Found in %.2f seconds" % (end-start)), iters, "iterations"
print ("Centroids:")
print (centers)
print ("Cluster assignments:", assignments)
print("points")
print(points[1])
'''
data = {"x": [], "y": [], "cluster": []}
for i in range(len(assignments)):
  data["x"].append(points[i][0])
  data["y"].append(points[i][1])
  data["cluster"].append(assignments[i])
df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, 
           fit_reg=False, size=7, 
           hue="cluster", legend=False)
plt.show()
'''
