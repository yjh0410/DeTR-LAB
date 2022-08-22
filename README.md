# DeTR-LAB
Library of Detection with Transformer

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n detr python=3.6
```

- Then, activate the environment:
```Shell
conda activate detr
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

We suggest that PyTorch should be higher than 1.9.0 and Torchvision should be higher than 0.10.3. 
At least, please make sure your torch is version 1.x.

# Experiments
## DeTR
| Model        |  backbone  | FPS<sup><br>2080ti |  FLOPs   |  Params |    AP    |  Weight  |
|--------------|------------|--------------------|----------|---------|----------|----------|
| DeTR-R50     |    R-50    |  37                |  95.2 B  |  36.7 M |   41.7   | [github]() |
| DeTR-R50-DC5 |    R-50    |  20                |  162.1 B |  48.3 M |   43.0   | [github]() |
| DeTR-R101    |    R-101   |  25                |  174.7 B |  55.7 M |   41.0   | [github]() |
| DeTR-R101-DC5|    R-101   |  14                |  241.6 B |  55.7 M |   41.0   | [github]() |
