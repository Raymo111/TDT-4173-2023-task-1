# Report - Raymond Li

## K-means

### How it works

The k-means algorithm is a classification algorithm that partitions n points in a dataset into k clusters in which each observation belongs to the cluster with the nearest centroid, the center of the cluster. The algorithm starts by selecting k points from the dataset as the initial centroids, such that the points are as far apart as possible. Then, the algorithm iterates between two steps until the centroids converge. In the first step, each point is assigned to the cluster belonging to the centroid with the smallest Euclidean distance to the point. In the second step, the centroids are updated to the mean Euclidean distance of all observations in the cluster. The algorithm is guaranteed to converge, but it may converge to a local optimum. To predict the cluster of a new point, the algorithm finds the centroid with the smallest Euclidean distance to the point.

### What problems it's suited for

The k-means algorithm is suited for problems where the data can be partitioned into k categories/clusters. Euclidean distance should be a good measure of similarity between observations, and the data should be continuous and separated by relatively clear boundaries.

### Inductive bias

The k-means algorithm assumes that the data can be partitioned into k spherical, similarly-sized clusters. The algorithm also assumes that the data is continuous and that Euclidean distance is a good measure of similarity between observations.

### Second problem challenges

In the second problem, the data's x0 was significantly different in magnitude to its x1, resulting in the Euclidean distance between observations being dominated by the difference in x0 and negatively impacting the partitioning of the data.

This violates the indutive bias of the k-means algorithm, as it assumes that Euclidean distance is a good measure of similarity between data points, and in this case it was not.

#### Modifications

I rescaled the x0 of all points by dividing it by 10, which normalized the Euclidean distance between the axes of points, solving the axis-domination problem, before running the k-means algorithm. This enabled the algorithm to partition the data into the correct clusters.

## Decision Tree

### How it works

The decision tree algorithm is a classification algorithm that builds a tree of decisions based on the features of the data. The algorithm repeatedly finds a feature that best splits the data into two groups, based on entropy, and splits on that feature. The algorithm continues to split the data until all observations in a node belong to the same class, or the node contains less than a minimum number of observations, or some other hyperparameter is fulfilled. To predict the class of a new point, the algorithm traverses the tree based on the features of the point until it reaches a leaf node, and returns the class of that node.

### What problems it's suited for

The decision tree algorithm is suited for problems where the data can be split into categories based on the features of the data. The data should be discrete, and the features should be categorical.

### Inductive bias

The decision tree algorithm assumes that the data can be split into categories based on the features of the data. Trees should be shorter and place high information gain features closer to the root.

### Second problem challenges

In the second problem, the validation and test sets contain attribute combinations not present in the training data. This was due to the Birth Month attribute splitting the data too irrelevantly, negatively impacting the accuracy.

#### Modifications

I preprocessed the data by removing the Birth Month attribute, which was not really relevant to the final result.
