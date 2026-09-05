Optimizing performance of hand-coded neural net.

**Benchmark hardware**

| Spec | Windows | Mac |
| --- | --- | --- |
| CPU | Intel Core i7-10700K @ 3.80 GHz | Apple M1 Pro @ 3.2 GHz |
| Cores / logical processors | 8 / 16 | 10 / 10 |

**Network**

- Architecture: 784 -> 256 -> 256 -> 10, sigmoid activations
- Loss: half squared error
- Learning rate: 1.5
- Batch size: 10
- Epochs: 30
- Dataset: MNIST
- Training images: 60,000
- Initial accuracy: 99.80% training; 98.28% test

<table>
  <thead>
    <tr>
      <th rowspan="2">Version</th>
      <th colspan="2" align="center">Windows</th>
      <th colspan="2" align="center">Mac</th>
    </tr>
    <tr>
      <th>Training time</th>
      <th>Inference time (10k images)</th>
      <th>Training time</th>
      <th>Inference time (10k images)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>initial naive implementation (c++)</td><td>685.3s</td><td>2.257s</td><td>577.5s</td><td>2.086s</td></tr>
    <tr><td>loop reordering</td><td>565.3s</td><td>2.257s</td><td>437.3s</td><td>2.089s</td></tr>
    <tr><td>avx2/neon intrinsics for forward pass dot products</td><td>223.8s</td><td>0.338s</td><td>171.4s</td><td>0.597s</td></tr>
    <tr><td>avx2/neon intrinsics for layer 1 gradient calc</td><td>133.7s</td><td>0.338s</td><td>174.6s</td><td>0.597s</td></tr>
    <tr><td>extend avx2 calcs to 4/8-way</td><td>116.2s</td><td>0.290s</td><td></td><td></td></tr>
  </tbody>
</table>
