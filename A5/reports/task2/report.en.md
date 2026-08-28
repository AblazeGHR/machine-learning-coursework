# Task 2: CNN Autoencoder

## 1. Task Objective

The goal of this task is to build a CNN-based autoencoder for CIFAR-10 and use a 4-dimensional latent space to compress and reconstruct images.

The task requirements are:

- Use a CNN architecture for both the encoder and decoder
- Set the latent dimension to 4
- Reconstruct images after training
- Compare the results with the Task 1 MLP Autoencoder

## 2. Model Architecture

The model is implemented in `src/models/task2_cnn_autoencoder.py`.

### Encoder
- Input: `3 x 32 x 32`
- `Conv2d(3 -> 32, kernel=3, stride=2, padding=1)` + ReLU
- `Conv2d(32 -> 64, kernel=3, stride=2, padding=1)` + ReLU
- `Conv2d(64 -> 128, kernel=3, stride=2, padding=1)` + ReLU
- Flatten
- `Linear(128 * 4 * 4 -> 4)`

### Decoder
- `Linear(4 -> 128 * 4 * 4)` + ReLU
- Reshape to `128 x 4 x 4`
- `ConvTranspose2d(128 -> 64, kernel=4, stride=2, padding=1)` + ReLU
- `ConvTranspose2d(64 -> 32, kernel=4, stride=2, padding=1)` + ReLU
- `ConvTranspose2d(32 -> 16, kernel=4, stride=2, padding=1)` + ReLU
- `Conv2d(16 -> 3, kernel=3, stride=1, padding=1)` + Sigmoid

## 3. Training Setup

The training script is located at `scripts/train_task2.py`.

- Dataset: CIFAR-10
- Data loader: `src/data/cifar10.py`
- Loss: `MSELoss`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Batch size: `128` by default
- Latent dimension: `4`
- Device: automatically detects `cuda` / `cpu`
- Output directory: `reports/task2/`
  - `data/` stores checkpoints and training history
  - `figures/` stores reconstruction visualizations

## 4. Current Results

I completed a smoke test and then ran a formal training session.

### Smoke Test Command

```powershell
python .\scripts\train_task2.py --epochs 1 --batch-size 64 --max-train-batches 2 --max-test-batches 1
```

### Results

- `train_loss = 0.060081`
- `test_loss = 0.064947`

### Formal Training

The formal training was run on the full training and test sets with the following command:

```powershell
python .\scripts\train_task2.py --epochs 3 --batch-size 256 --seed 42
```

### Formal Training Results

- `epoch 1: train_loss = 0.039766, test_loss = 0.030985`
- `epoch 2: train_loss = 0.030711, test_loss = 0.030491`
- `epoch 3: train_loss = 0.030431, test_loss = 0.030284`

### Visualization Result

A preview of the original images and reconstructions was generated:

- `reports/task2/figures/task2_reconstruction_preview.png`

A loss curve was also generated:

- `reports/task2/figures/task2_loss_curve.png`

### Comparison with Task 1

Under the same number of epochs, batch size, and random seed settings, Task 1 reached a final test loss of `0.030224`, while Task 2 reached `0.030284`.

The two results are very close. In this run, Task 1 performed slightly better than Task 2, but the difference is very small, which suggests that the CNN Autoencoder and the MLP Autoencoder achieve reconstruction errors of the same order of magnitude under the current setup.

### MSE comparison (brief)

- Per-image reconstruction MSE was computed for both models on the full CIFAR-10 test set and the comparison outputs were saved under `reports/compare/`:
  - Summary: `reports/compare/task1_task2_mse_summary.json`
  - Per-image arrays: `reports/compare/task1_task2_per_image_mse.npz`
  - Figures: `reports/compare/figures/per_image_mse_histogram.png`, `reports/compare/figures/per_image_mse_diff_histogram.png`
- Key numbers (n=10000):
  - MLP mean MSE = 0.03022369
  - CNN mean MSE = 0.03028437
  - Mean difference (CNN - MLP) ≈ 0.0000607 (≈ 0.2% relative)
  - Per-image wins: MLP better on 5068 images, CNN better on 4932 images

Conclusion: The two models exhibit nearly identical reconstruction MSE on the test set. The tiny advantage of MLP is within run-to-run variability and may stem from randomness, short training (3 epochs), architectural differences, or limitations of MSE as a perceptual metric. For more decisive comparisons, run multiple seeds/longer training and consider perceptual metrics (PSNR/SSIM/LPIPS) and visual inspection of worst-case reconstructions.

## 5. Output Files

### Model and Training History

- `reports/task2/data/task2_cnn_autoencoder.pt`
- `reports/task2/data/task2_history.json`

### Figures

- `reports/task2/figures/task2_reconstruction_preview.png`
- `reports/task2/figures/task2_loss_curve.png`

## 6. Future Extensions

The following extensions can be completed later:

1. Train for more epochs
2. Compare the Task 2 results with the Task 1 MLP Autoencoder
3. Measure distances between different classes in the embedding space
4. Try different latent dimensions and observe changes in test error
5. Add more quantitative metrics and deeper analysis

