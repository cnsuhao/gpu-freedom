#!/usr/bin/python3
# code from book "First Contact with Tensorflow" by Jordi Torres
import tensorflow as tf
import numpy as np

num_points=2000
vectors_set=[]

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0,0.9),
                            np.random.normal(0.0,0.9)])
    else:
        vectors_set.append([np.random.normal(3.0,0.5),
                            np.random.normal(1.0,0.5)])

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"x": [v[0] for v in vectors_set],
                   "y": [v[1] for v in vectors_set]})
#sns.lmplot("x","y",data=df, fit_reg=False, size=6)
#plt.show()

# Code presented in blog by Shawn Simister
vectors=tf.constant(vectors_set)
k=4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))
# dm: it is necessary to initialize update_centroides, as assign statement
#     below does not work
update_centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

expanded_vectors=tf.expand_dims(vectors,0)
expanded_centroides=tf.expand_dims(centroides,1)

print(expanded_vectors.get_shape())
print(expanded_centroides.get_shape())

distances = tf.reduce_sum(
  tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2)
print(distances.get_shape())

assignments = tf.argmin(distances, 0)
print(assignments.get_shape())



means = tf.concat(0,[tf.reduce_mean(tf.gather(vectors,
                  tf.reshape(tf.where(tf.equal(assignments, c)),[-1,1])),
                  reduction_indices=[1]) for c in range(k)])

print("shape of means:")
print(means.get_shape())
# not worky
#update_centroides=tf.assign(centroides, means)

tf.assign(centroides, means)
tf.assign(update_centroides, centroides)


init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)
print("session:")
#print(sess.run([centroides, assignments]))
print(sess.run([update_centroides, centroides, assignments]))

#print(update_centroides)

for step in range(100):
    (_,centroid_values,assignment_values) = sess.run([update_centroides,
                                                    centroides, assignments])

print("centroids")
print(centroid_values)

data = { "x": [], "y": [], "cluster": []}


for i in range(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x","y",data=df, fit_reg=False, size=6, hue="cluster", legend=False)

plt.show()

        
