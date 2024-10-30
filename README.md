## Candle 2683 minrep

Minimal reproducible example for: https://github.com/huggingface/candle/issues/2583.

## Results 

```nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
name, compute_cap, driver_version
NVIDIA GeForce RTX 4090, 8.9, 550.107.02
```

| batch_size | tch_μs_average | cdl_μs_average |
|------------|----------------|----------------|
|          1 |              8 |              6 |
|          2 |              8 |              7 |
|          4 |              8 |              8 |
|          8 |              9 |             10 |
|         16 |             11 |             16 |
|         32 |             16 |             26 |
|         64 |             13 |             46 |
|        128 |             19 |             87 |
|        256 |             44 |            192 |
|        512 |             54 |            338 |
|       1024 |            102 |            693 |
|       2048 |            208 |           1457 |
