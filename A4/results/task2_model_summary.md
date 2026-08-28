# Task 2 Model Summary

## Architecture

- Layer 0: `Linear(in_features=561, out_features=256, bias=True)`
- Layer 1: `ReLU()`
- Layer 2: `Dropout(p=0.2, inplace=False)`
- Layer 3: `Linear(in_features=256, out_features=128, bias=True)`
- Layer 4: `ReLU()`
- Layer 5: `Dropout(p=0.2, inplace=False)`
- Layer 6: `Linear(in_features=128, out_features=6, bias=True)`

## Trainable Parameters

- `net.0.weight`: 143616
- `net.0.bias`: 256
- `net.3.weight`: 32768
- `net.3.bias`: 128
- `net.6.weight`: 768
- `net.6.bias`: 6

- **Total trainable parameters**: 177542
