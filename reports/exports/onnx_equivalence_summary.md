# Combined ONNX Export & Output Equivalence Summary

# Numerical Equivalence Report: AGRICULTURE
- **PyTorch Model:** `Agriculture_bestweight.pt`
- **ONNX Model:** `agriculture.onnx`
- **Tolerance Threshold:** `<= 0.001`
- **Status:** **PASS (Equivalence Confirmed)**
- **Overall Max Absolute Error:** `3.662109e-04`
- **Overall Mean Absolute Error:** `8.252605e-06`

## Sample Level Verification

| Sample # | Input Source | Shape | Max Abs Error | Mean Abs Error | Status |
|---|---|:---:|:---:|:---:|:---:|
| 1 | `dead_crop_0059.png` | [1, 3, 512, 512] | `2.441406e-04` | `8.422430e-06` | **PASS** |
| 2 | `dead_crop_0064.png` | [1, 3, 512, 512] | `2.441406e-04` | `8.052049e-06` | **PASS** |
| 3 | `dead_crop_0078.png` | [1, 3, 512, 512] | `3.356934e-04` | `8.369484e-06` | **PASS** |
| 4 | `dead_crop_0094.png` | [1, 3, 512, 512] | `3.662109e-04` | `8.479000e-06` | **PASS** |
| 5 | `dead_crop_0109.png` | [1, 3, 512, 512] | `3.356934e-04` | `8.238616e-06` | **PASS** |
| 6 | `Synthetic Random Tensor` | [1, 3, 512, 512] | `3.051758e-04` | `8.195389e-06` | **PASS** |
| 7 | `Synthetic Random Tensor` | [1, 3, 512, 512] | `2.746582e-04` | `8.107846e-06` | **PASS** |
| 8 | `Synthetic Random Tensor` | [1, 3, 512, 512] | `2.441406e-04` | `8.097845e-06` | **PASS** |
| 9 | `Synthetic Random Tensor` | [1, 3, 512, 512] | `3.662109e-04` | `8.296660e-06` | **PASS** |
| 10 | `Synthetic Random Tensor` | [1, 3, 512, 512] | `3.356934e-04` | `8.266735e-06` | **PASS** |
---

# Numerical Equivalence Report: SHIPS
- **PyTorch Model:** `best.pt`
- **ONNX Model:** `ships.onnx`
- **Tolerance Threshold:** `<= 0.001`
- **Status:** **PASS (Equivalence Confirmed)**
- **Overall Max Absolute Error:** `6.408691e-04`
- **Overall Mean Absolute Error:** `1.322155e-05`

## Sample Level Verification

| Sample # | Input Source | Shape | Max Abs Error | Mean Abs Error | Status |
|---|---|:---:|:---:|:---:|:---:|
| 1 | `cargo_ship_0041.png` | [1, 3, 640, 640] | `5.493164e-04` | `1.319741e-05` | **PASS** |
| 2 | `cargo_ship_0048.png` | [1, 3, 640, 640] | `5.493164e-04` | `1.426493e-05` | **PASS** |
| 3 | `cargo_ship_0087.png` | [1, 3, 640, 640] | `5.493164e-04` | `1.366991e-05` | **PASS** |
| 4 | `cargo_ship_0091.png` | [1, 3, 640, 640] | `4.882812e-04` | `1.433476e-05` | **PASS** |
| 5 | `cargo_ship_0105.png` | [1, 3, 640, 640] | `4.882812e-04` | `1.280871e-05` | **PASS** |
| 6 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `5.493164e-04` | `1.279676e-05` | **PASS** |
| 7 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `6.408691e-04` | `1.277110e-05` | **PASS** |
| 8 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `4.272461e-04` | `1.289291e-05` | **PASS** |
| 9 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `4.272461e-04` | `1.280410e-05` | **PASS** |
| 10 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `4.577637e-04` | `1.267486e-05` | **PASS** |
---

# Numerical Equivalence Report: SPACE_ELECTRONICS
- **PyTorch Model:** `space_electronics.pt`
- **ONNX Model:** `space_electronics.onnx`
- **Tolerance Threshold:** `<= 0.001`
- **Status:** **PASS (Equivalence Confirmed)**
- **Overall Max Absolute Error:** `5.187988e-04`
- **Overall Mean Absolute Error:** `1.303122e-05`

## Sample Level Verification

| Sample # | Input Source | Shape | Max Abs Error | Mean Abs Error | Status |
|---|---|:---:|:---:|:---:|:---:|
| 1 | `open_0060.png` | [1, 3, 640, 640] | `4.272461e-04` | `1.418791e-05` | **PASS** |
| 2 | `open_0100.png` | [1, 3, 640, 640] | `4.272461e-04` | `1.220561e-05` | **PASS** |
| 3 | `open_0106.png` | [1, 3, 640, 640] | `4.577637e-04` | `1.331133e-05` | **PASS** |
| 4 | `open_0110.png` | [1, 3, 640, 640] | `5.187988e-04` | `1.457502e-05` | **PASS** |
| 5 | `open_0116.png` | [1, 3, 640, 640] | `4.577637e-04` | `1.300621e-05` | **PASS** |
| 6 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `4.882812e-04` | `1.261197e-05` | **PASS** |
| 7 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `4.577637e-04` | `1.243831e-05` | **PASS** |
| 8 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `4.577637e-04` | `1.271325e-05` | **PASS** |
| 9 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `5.187988e-04` | `1.279249e-05` | **PASS** |
| 10 | `Synthetic Random Tensor` | [1, 3, 640, 640] | `4.425049e-04` | `1.247008e-05` | **PASS** |
---
