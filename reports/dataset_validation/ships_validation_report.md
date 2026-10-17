# Dataset Pre-Flight Validation Report: SHIPS
- **Path:** `D:\Accion_Lab\Synthline\environmental defense\ship_final_dataset`
- **Allowed Classes:** {0: 'cargo_ship', 1: 'passenger_ship'}
- **Overall Training Sanity:** PASS
- **Total Images:** 883
- **Total Labels:** 767
- **Total Bounding Boxes:** 767
- **Pseudo-Boxes Flagged:** 767

## Split Breakdown

| Split | Images | Labels | Corrupt Imgs | Corrupt Lbls | Invalid Boxes | Pseudo Boxes (%) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `train` | 591 | 489 | 0 | 0 | 0 | 489 (100.0%) | **PASS** |
| `val` | 213 | 198 | 0 | 0 | 0 | 198 (100.0%) | **PASS** |
| `test` | 79 | 80 | 0 | 0 | 0 | 80 (100.0%) | **PASS** |

## Class Distribution by Split

- **train:** Class 0 (cargo_ship): 92, Class 1 (passenger_ship): 397
- **val:** Class 0 (cargo_ship): 118, Class 1 (passenger_ship): 80
- **test:** Class 0 (cargo_ship): 39, Class 1 (passenger_ship): 41