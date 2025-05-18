# EEE474-Project
# Unsupervised Handwritten Digit Clustering with Autoencoder and SOM

This project implements an unsupervised learning approach for clustering handwritten digits using a combination of deep learning and neural clustering techniques. The main goal is to group digit images without using labels and evaluate the clustering performance.

## Project Overview

We use the following pipeline:

- Flattened grayscale images of handwritten digits (e.g., MNIST-like)
- Feature extraction using an **autoencoder**
- Clustering using **Kohonen Self-Organizing Map (SOM)** and **KMeans**
- Visualization using **t-SNE**
- Evaluation using clustering metrics with true digit labels

##  Model Architecture

### Autoencoder
Used for dimensionality reduction:

- **Encoder**: Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU)
- **Decoder**: Dense(128, ReLU) → Dense(256, ReLU) → Dense(784, Sigmoid)

### Clustering

- **KMeans**: Standard unsupervised clustering.
- **MiniSom**: A 2D Self-Organizing Map (Kohonen Network) for topological clustering.

##  Dataset

- Image-based dataset of handwritten digits.
- Images are preprocessed and flattened into 1D vectors.
- Labels are used only for clustering performance evaluation.

##  Evaluation Metrics

Since the task is unsupervised, we use the following clustering metrics:

- Silhouette Score
- Homogeneity Score
- Completeness Score
- V-Measure
- Adjusted Rand Index (ARI)
- Adjusted Mutual Information (AMI)

##  Visualizations

- 2D t-SNE projection of encoded vectors.
- Cluster visualization colored by KMeans/SOM predictions.

##  Limitations

- No dropout or batch normalization applied.
- MiniSom has randomness and can yield variable results.
- Limited parameter search.

## 🛠 Requirements

```bash
pip install tensorflow scikit-learn minisom seaborn matplotlib pandas
