# CIFAR-10 Autoencoder Final Report

## 1. Project Overview

This project completed four stages of autoencoder experiments on CIFAR-10:

- Task 1: MLP Autoencoder, latent dimension = 4
- Task 2: CNN Autoencoder, latent dimension = 4
- Task 3: Class-distance analysis in the CNN autoencoder embedding space
- Task 4: Latent dimension sweep to study how test error changes with embedding size

### Overall Conclusions

- Task 1 and Task 2 both completed full reconstruction training and produced stable reconstruction figures and loss curves
- Task 1 additionally produced the required 800-sample 2D embedding scatter plot for class-clustering inspection
- Task 3 shows that the 4D embedding already contains meaningful semantic structure and can be used for class-distance analysis
- Task 4 shows that larger latent dimensions lead to lower reconstruction error; `latent_dim=16` performed best in the current sweep

## 2. Experimental Setup

### Shared Setup

- Dataset: CIFAR-10
- Loss: MSELoss
- Optimizer: Adam
- Learning rate: `1e-3`
- Random seed: `42`
- Training and evaluation are based on `src/data/cifar10.py`

### Task Breakdown

| Task | Model | Latent dim | Main Goal |
|---|---|---:|---|
| Task 1 | MLP Autoencoder | 4 | Verify MLP reconstruction ability |
| Task 2 | CNN Autoencoder | 4 | Compare reconstruction performance with Task 1 |
| Task 3 | CNN Embedding Analysis | 4 | Analyze class distances and interpretability |
| Task 4 | CNN Sweep | 1 / 2 / 4 / 8 / 16 | Study test error vs latent dimension |

## 3. Task 1: MLP Autoencoder

### Results Summary

Task 1 used an MLP encoder/decoder with latent dimension 4.

Formal training results:

- Epoch 1: train loss = `0.038540`, test loss = `0.032669`
- Epoch 2: train loss = `0.030928`, test loss = `0.030403`
- Epoch 3: train loss = `0.030367`, test loss = `0.030224`

### Scatter Plot Generation Steps and Rationale

1. First, I loaded the trained Task 1 checkpoint, because the goal here is to analyze the trained model rather than retrain it.
2. Then, I randomly sampled 800 images from the CIFAR-10 test set, because the assignment explicitly requests a random subset and the test set is better for checking the learned representation after training.
3. Next, I passed the images through the encoder to obtain 4D latent vectors and projected them to 2D with PCA, because the assignment requires a 2D embedding plane even though the latent space is 4D.
4. After that, I colored the scatter plot by class label, because color makes it easy to inspect whether same-class samples cluster together.
5. Finally, I saved the figure and intermediate data under `reports/task1/`, because that keeps the result reproducible and easy to verify later.

### Figure Placeholders

- [Figure placeholder: Task 1 original vs reconstruction]
- [Figure placeholder: Task 1 loss curve]
- [Figure placeholder: Task 1 800-sample 2D embedding scatter plot]

### Output Files

- `reports/task1/data/task1_mlp_autoencoder.pt`
- `reports/task1/data/task1_history.json`
- `reports/task1/data/task1_embedding_scatter.npz`
- `reports/task1/data/task1_embedding_scatter.json`
- `reports/task1/figures/task1_reconstruction_preview.png`
- `reports/task1/figures/task1_loss_curve.png`
- `reports/task1/figures/task1_embedding_scatter.png`

## 4. Task 2: CNN Autoencoder

### Results Summary

Task 2 used a CNN encoder/decoder with latent dimension 4.

Formal training results:

- Epoch 1: train loss = `0.039766`, test loss = `0.030985`
- Epoch 2: train loss = `0.030711`, test loss = `0.030491`
- Epoch 3: train loss = `0.030431`, test loss = `0.030284`

### Comparison with Task 1

- Task 1 final test loss: `0.030224`
- Task 2 final test loss: `0.030284`

The two results are very close, which means the MLP Autoencoder and CNN Autoencoder achieve reconstruction errors of the same order of magnitude under the current setup.

### Figure Placeholders

- [Figure placeholder: Task 2 original vs reconstruction]
- [Figure placeholder: Task 2 loss curve]

### Output Files

- `reports/task2/data/task2_cnn_autoencoder.pt`
- `reports/task2/data/task2_history.json`
- `reports/task2/figures/task2_reconstruction_preview.png`
- `reports/task2/figures/task2_loss_curve.png`

## 5. Task 3: Embedding Distance Analysis

### Results Summary

Task 3 analyzed class centers in the 4D embedding learned by Task 2 on the CIFAR-10 test set.

Key results:

- Globally nearest pair: `deer` and `frog`, distance `0.4133`
- Globally farthest pair: `airplane` and `frog`, distance `5.4487`
- Embedding dimension: `4`
- Number of analyzed samples: `10000`

### Conclusion

The 4D embedding already exhibits a meaningful class structure: semantically similar classes are placed closer together, while semantically different classes are farther apart.

### Figure Placeholders

- [Figure placeholder: Task 3 distance heatmap]
- [Figure placeholder: Task 3 class-center distance bar chart]

### Output Files

- `reports/task3/data/task3_embeddings.npz`
- `reports/task3/data/task3_class_centers.json`
- `reports/task3/data/task3_distance_matrix.csv`
- `reports/task3/data/task3_distance_summary.json`
- `reports/task3/data/task3_config.json`
- `reports/task3/figures/task3_distance_heatmap.png`
- `reports/task3/figures/task3_center_distance_bar.png`

## 6. Task 4: Latent Dimension Sweep

### Results Summary

Task 4 swept latent dimensions `1, 2, 4, 8, 16` and compared the final test loss.

| latent_dim | final_test_loss |
|---:|---:|
| 1  | 0.0430095438 |
| 2  | 0.0368811189 |
| 4  | 0.0302826319 |
| 8  | 0.0244705736 |
| 16 | 0.0183803936 |

### Conclusion

- Larger latent dimensions lead to lower test error
- The best dimension in the current sweep is `16`
- `latent_dim=4` remains a compact and interpretable middle ground

### Figure Placeholders

- [Figure placeholder: Task 4 test error vs latent dimension]
- [Figure placeholder: Task 4 loss curves]
- [Figure placeholder: Task 4 latent_dim=4 reconstruction]
- [Figure placeholder: Task 4 latent_dim=16 reconstruction]

### Output Files

- `reports/task4/data/task4_sweep_config.json`
- `reports/task4/data/task4_sweep_results.csv`
- `reports/task4/data/task4_sweep_results.json`
- `reports/task4/data/task4_latent_1.pt`
- `reports/task4/data/task4_latent_2.pt`
- `reports/task4/data/task4_latent_4.pt`
- `reports/task4/data/task4_latent_8.pt`
- `reports/task4/data/task4_latent_16.pt`
- `reports/task4/figures/task4_test_error_vs_latent_dim.png`
- `reports/task4/figures/task4_loss_curves.png`

## 7. Cross-Task Comparison

| Task | Main Question | Key Finding |
|---|---|---|
| Task 1 | Can an MLP reconstruct CIFAR-10? | Yes, with stable convergence |
| Task 2 | Is CNN better than MLP? | Their reconstruction errors are very close |
| Task 3 | Does the 4D embedding have structure? | Yes, class distances are meaningful |
| Task 4 | How does latent dim affect error? | Larger is better; 16D is best |

## 8. Final Summary

This project completed a full autoencoder workflow from reconstruction, comparison, representation analysis, to latent-dimension sweep.

The main takeaways are:

1. Autoencoders can compress and reconstruct CIFAR-10 images effectively
2. In the current setup, CNN and MLP reconstruction performance are similar
3. A 4D latent space already has interpretability value
4. If the goal is minimizing reconstruction error, a larger latent dimension is preferable

## 9. Appendix: Main Report Files

- `reports/task1/report.en.md`
- `reports/task2/report.en.md`
- `reports/task3/report.en.md`
- `reports/task4/report.en.md`

