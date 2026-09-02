# Optimizing hand-coded neural net from scratch.

## Benchmark hardware

- CPU: Intel Core i7-10700K @ 3.80 GHz
- Cores / logical processors: 8 / 16

## Fixed training parameters

- Architecture: 784 -> 256 -> 256 -> 10, sigmoid activations
- Loss: half squared error
- Learning rate: 1.5
- Batch size: 10
- Epochs: 30
- Dataset: MNIST
- Training images: 60,000
- Inference images: 10,000
- Initial accuracy: 99.80% training; 98.28% test

| Version | Training time | Inference time |
| --- | ---: | ---: |
| Initial naive implementation | 685.3s | 2.257s |
