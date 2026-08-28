# Task 1: MLP Autoencoder

## 1. Task Objective

The goal of this task is to build an MLP-based autoencoder for CIFAR-10 and use a 4-dimensional latent space to compress and reconstruct images.

The task requirements are:

- Use an MLP architecture for both the encoder and decoder
- Set the latent dimension to 4
- Reconstruct images after training
- Visualize 800 random samples in a 2D embedding scatter plot colored by class label

## 2. Model Architecture

The model is implemented in `src/models/task1_mlp_autoencoder.py`.

### Encoder
- Input: `3 x 32 x 32`
- Flatten
- `Linear(3072 -> 1024)` + ReLU
- `Linear(1024 -> 256)` + ReLU
- `Linear(256 -> 64)` + ReLU
- `Linear(64 -> 4)`

### Decoder
- `Linear(4 -> 64)` + ReLU
- `Linear(64 -> 256)` + ReLU
- `Linear(256 -> 1024)` + ReLU
- `Linear(1024 -> 3072)` + Sigmoid
- Reshape back to `3 x 32 x 32`

## 3. Training Setup

The training script is located at `scripts/train_task1.py`.

- Dataset: CIFAR-10
- Data loader: `src/data/cifar10.py`
- Loss: `MSELoss`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Batch size: `128` by default
- Latent dimension: `4`
- Device: automatically detects `cuda` or `cpu`
- Output directory: `reports/task1/`
  - `data/` stores checkpoints and training history
  - `figures/` stores reconstruction visualizations

## 4. Current Results

I completed a smoke test and then ran a formal training session.

### Smoke Test Command

```powershell
python .\scripts\train_task1.py --epochs 1 --batch-size 64 --max-train-batches 2 --max-test-batches 1
```

### Results

- `train_loss = 0.056055`
- `test_loss = 0.063331`

### Formal Training

The formal training was run on the full training and test sets with the following command:

```powershell
python .\scripts\train_task1.py --epochs 3 --batch-size 256 --seed 42
```

### Formal Training Results

- `epoch 1: train_loss = 0.038540, test_loss = 0.032669`
- `epoch 2: train_loss = 0.030928, test_loss = 0.030403`
- `epoch 3: train_loss = 0.030367, test_loss = 0.030224`

### Visualization Result

A preview of the original images and reconstructions was generated:

- `reports/task1/figures/task1_reconstruction_preview.png`

A loss curve was also generated:

- `reports/task1/figures/task1_loss_curve.png`

In addition, the 800-sample 2D embedding scatter plot was generated to inspect class clustering:

- `reports/task1/figures/task1_embedding_scatter.png`

### Scatter Plot Generation Steps and Rationale

1. First, I loaded the model parameters from the trained Task 1 checkpoint, because this step is for analyzing the trained model rather than retraining it.
2. Then, I randomly sampled 800 images from the CIFAR-10 test set, because the assignment explicitly asks for a random subset and the test set is better for inspecting the learned representation after training.
3. Next, I passed each image through the encoder to obtain a 4D latent vector and used PCA to project it into 2D, because the original embedding is 4-dimensional while the assignment requires a 2D embedding plane.
4. After that, I colored the scatter plot by class label, because color makes it easy to see whether samples from the same class cluster together.
5. Finally, I saved the figure and intermediate results under `reports/task1/figures/` and `reports/task1/data/`, because this keeps the result reproducible and makes the report easy to verify later.

## 5. Output Files

### Model and Training History

- `reports/task1/data/task1_mlp_autoencoder.pt`
- `reports/task1/data/task1_history.json`

### Figures

- `reports/task1/figures/task1_reconstruction_preview.png`
- `reports/task1/figures/task1_loss_curve.png`

### Data

- `reports/task1/data/task1_embedding_scatter.npz`
- `reports/task1/data/task1_embedding_scatter.json`

## 6. Future Extensions

The following extensions can be completed later:

1. Train for more epochs
2. Compare the Task 1 MLP Autoencoder with the Task 2 CNN Autoencoder
3. Try t-SNE / UMAP for additional embedding visualization
4. Try different latent dimensions and observe changes in test error
5. Add more quantitative metrics and deeper analysis

