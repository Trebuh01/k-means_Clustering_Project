# k-means Clustering Project

## Overview

This project implements the k-means clustering algorithm from scratch in Python. The implementation includes both the standard k-means algorithm and the k-means++ initialization method to improve the clustering results. The project applies the algorithm to the Iris dataset to demonstrate its functionality.

## Features

- **Custom k-means Implementation:** Core k-means clustering logic without relying on external libraries like Scikit-learn.
- **k-means++ Initialization:** Enhanced centroid initialization for better convergence.
- **Evaluation:** Analysis of cluster assignments and intra-class variance.

## Files

- `k_means.py`: Contains the implementation of the k-means clustering algorithm, including the k-means++ initialization method.
- `main.py`: Main script to load the Iris dataset, perform clustering, and evaluate the results.


## Code Explanation

### `main.py`

- **Loading Data:**

    ```python
    def load_iris():
        data = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
        ...
        return features, classes
    ```

- **Clustering and Evaluation:**

    ```python
    def clustering(kmeans_pp):
        ...
        for i in range(100):
            assignments, centroids, error = k_means(features, 3, kmeans_pp)
            evaluate(assignments, classes)
            intra_class_variance.append(error)
        print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")
    ```

### `k_means.py`

- **Initialization Methods:**

    ```python
    def initialize_centroids_forgy(data, k):
        indices = np.random.choice(data.shape[0], k, replace=False)
        return data[indices, :]
    ```

    ```python
    def initialize_centroids_kmeans_pp(data, k):
        centroids = [data[np.random.randint(data.shape[0])]]
        ...
        return np.array(centroids)
    ```

- **Main k-means Logic:**

    ```python
    def k_means(data, num_centroids, kmeansplusplus=False):
        ...
        return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
    ```


