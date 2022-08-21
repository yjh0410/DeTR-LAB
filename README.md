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
| Model        |  backbone  | FPS<sup><br>2080ti |  FLOPs  |  Params |    AP    |    AP50    |  Weight  |
|--------------|------------|--------------------|---------|---------|----------|------------|----------|
| DeTR-R50     |    R-50    |  74                |  87.7 B |  48.3 M |   41.0   |    61.3    | [github]() |
| DeTR-R50-DC5 |    R-50    |  74                |  87.7 B |  48.3 M |   41.0   |    61.3    | [github]() |
| DeTR-R101    |    R-101   |  74                |  87.7 B |  48.3 M |   41.0   |    61.3    | [github]() |
| DeTR-R101-DC5|    R-101   |  74                |  87.7 B |  48.3 M |   41.0   |    61.3    | [github]() |
