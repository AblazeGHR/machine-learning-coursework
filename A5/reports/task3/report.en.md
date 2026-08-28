# Task 3: Embedding Distance Analysis of the CNN Autoencoder

## 1. Task Objective

The goal of this task is to analyze the class structure of the CNN autoencoder trained in Task 2, measure the distances between CIFAR-10 classes in the embedding space, and answer the following questions:

- Which classes are closest in the embedding space?
- Which classes are farthest apart?
- What does the class distance matrix look like?

This task does not retrain the model. Instead, it directly analyzes the checkpoint from Task 2.

## 2. Method

This task uses the trained model checkpoint `reports/task2/data/task2_cnn_autoencoder.pt`, loads the `CNNAutoencoder` defined in `src/models/task2_cnn_autoencoder.py`, and calls `encode()` to extract 4-dimensional latent representations for each image in the CIFAR-10 test set.

The analysis pipeline is as follows:

1. Use the CIFAR-10 test split for analysis
2. Extract a 4D embedding for every sample
3. Compute the mean embedding per class to obtain class centers
4. Compute Euclidean distances between class centers to form a 10 × 10 distance matrix
5. Find the globally nearest and farthest class pairs
6. Summarize the nearest and farthest class for each category

The analysis script is located at `scripts/analyze_task3.py`.

## 3. Results Summary

This analysis processed `10000` test samples, used an embedding dimension of `4`, and covered all `10` classes.

### Globally Nearest Pair

- `deer` and `frog`
- Distance: `0.4133`

### Globally Farthest Pair

- `airplane` and `frog`
- Distance: `5.4487`

### Observations from the Distance Matrix

From the distance matrix and heatmap, we can see that:

- `deer`, `frog`, `cat`, and `dog` tend to be closer to each other in the embedding space.
- `airplane` is relatively far from most classes, especially from `frog`.
- `ship` and `truck` are also close, indicating that these two classes are nearby in the latent space.

## 4. Output Files

### Data Files

- `reports/task3/data/task3_embeddings.npz`
- `reports/task3/data/task3_class_centers.json`
- `reports/task3/data/task3_distance_matrix.csv`
- `reports/task3/data/task3_distance_summary.json`
- `reports/task3/data/task3_config.json`

### Figures

- `reports/task3/figures/task3_distance_heatmap.png`
- `reports/task3/figures/task3_center_distance_bar.png`

## 5. Conclusion

Task 3 shows that the 4D embedding learned by the CNN autoencoder is not random. Instead, it already captures a meaningful structure in which semantically similar classes are mapped closer together.

In particular:

- `deer` and `frog` are the closest pair
- `airplane` and `frog` are the farthest pair

This suggests that the latent space is interpretable and can serve as a useful basis for further analysis.

## 6. Future Extensions

The following extensions can be completed later:

1. Visualize the embeddings with PCA or t-SNE
2. Compare the class centers on the train split and the test split
3. Compare the Task 3 distance analysis with Task 1 embeddings
4. Repeat the analysis for other latent dimensions

