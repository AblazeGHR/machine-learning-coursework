# Task 4: Latent Dimension Sweep for the CNN Autoencoder

## 1. Task Objective

The goal of this task is to repeat the Task 2 CNN autoencoder experiment while varying the embedding / latent dimension, and observe how the test error changes as the latent dimension increases.

This task answers the following questions:

- Does the reconstruction error decrease when the latent dimension becomes larger?
- Which latent dimension performs best under the current setup?
- How does the 4D embedding compare with larger latent spaces?

## 2. Experimental Setup

This task uses the same CNN autoencoder architecture and data pipeline as Task 2, while only changing the latent dimension.

### Model

- Model: `CNNAutoencoder` defined in `src/models/task2_cnn_autoencoder.py`
- Loss function: `MSELoss`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Batch size: `256`
- Epochs: `3`
- Seed: `42`

### Swept latent dimensions

The following latent dimensions were evaluated:

- `1`
- `2`
- `4`
- `8`
- `16`

### Training Script

The experiment script is located at `scripts/train_task4.py`.

## 3. Experimental Results

### Summary Table

| latent_dim | final_train_loss | final_test_loss |
|-----------:|-----------------:|----------------:|
| 1  | 0.0433448335 | 0.0430095438 |
| 2  | 0.0368676088 | 0.0368811189 |
| 4  | 0.0304337269 | 0.0302826319 |
| 8  | 0.0246011597 | 0.0244705736 |
| 16 | 0.0198395957 | 0.0183803936 |

### Observations

The test loss decreases clearly as the latent dimension grows, which indicates that a larger embedding space can preserve more information and therefore improve reconstruction quality.

In particular:

- `latent_dim=1` gives the highest test loss and the weakest reconstruction ability
- `latent_dim=16` gives the lowest test loss and is the best result in this sweep
- `latent_dim=4` is consistent with the baseline used in Task 2 and Task 3, and sits in the middle of the range

### Trend of the Error

From `1 -> 2 -> 4 -> 8 -> 16`, the final test loss decreases almost monotonically:

- `0.0430 -> 0.0369 -> 0.0303 -> 0.0245 -> 0.0184`

This shows that latent dimension is an important factor for autoencoder reconstruction quality.

### Relation to Task 2 / Task 3

- Task 2 fixes the latent space at 4 dimensions
- Task 3 analyzes class distances based on the 4D embedding
- Task 4 further shows that although 4D already provides a meaningful structure, larger latent dimensions are better if the main goal is to minimize reconstruction error

## 4. Output Files

### Data Files

- `reports/task4/data/task4_sweep_config.json`
- `reports/task4/data/task4_sweep_results.csv`
- `reports/task4/data/task4_sweep_results.json`
- `reports/task4/data/task4_latent_1.pt`
- `reports/task4/data/task4_latent_2.pt`
- `reports/task4/data/task4_latent_4.pt`
- `reports/task4/data/task4_latent_8.pt`
- `reports/task4/data/task4_latent_16.pt`
- `reports/task4/data/task4_latent_1_history.json`
- `reports/task4/data/task4_latent_2_history.json`
- `reports/task4/data/task4_latent_4_history.json`
- `reports/task4/data/task4_latent_8_history.json`
- `reports/task4/data/task4_latent_16_history.json`

### Figures

- `reports/task4/figures/task4_test_error_vs_latent_dim.png`
- `reports/task4/figures/task4_loss_curves.png`
- `reports/task4/figures/task4_latent_1_loss_curve.png`
- `reports/task4/figures/task4_latent_2_loss_curve.png`
- `reports/task4/figures/task4_latent_4_loss_curve.png`
- `reports/task4/figures/task4_latent_8_loss_curve.png`
- `reports/task4/figures/task4_latent_16_loss_curve.png`
- `reports/task4/figures/task4_latent_4_reconstruction_preview.png`
- `reports/task4/figures/task4_latent_16_reconstruction_preview.png`

## 5. Conclusion

This experiment shows that the reconstruction error of the CNN autoencoder decreases as the latent dimension increases.

Under the current setup:

- The best latent dimension is `16`
- The corresponding final test loss is `0.0183803936`
- Compared with `latent_dim=1`, the test loss is significantly lower

Therefore, if the main objective is to minimize reconstruction error, a larger embedding dimension is more effective. However, if the goal is to obtain a compact and interpretable latent space, a 4D embedding is still valuable for analysis.

## 6. Future Extensions

The following extensions can be completed later:

1. Sweep more latent dimensions, such as `32` or `64`
2. Compare reconstruction image quality across latent dimensions
3. Combine this study with Task 3 to evaluate interpretability under different dimensions
4. Further investigate the relation between reconstruction error and class separability

