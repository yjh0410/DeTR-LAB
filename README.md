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

# This repository
In this repository, you can enjoy: 
- [x] DETR
- [x] Anchor-DETR
- [ ] DINO


# Experiments
## DETR

I evaluate DETR by loading official pretrained weight.

| Model        |  backbone  |  FPS<sup><br>3090  |  FLOPs   |  Params |    AP    |  Weight  |
|--------------|------------|--------------------|----------|---------|----------|----------|
| DETR-R50     |    R-50    |  37                |  95.2 B  |  36.7 M |   41.7   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r50-e632da11.pth) |
| DETR-R50-DC5 |    R-50    |  20                |  162.1 B |  36.7 M |   43.0   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r50-dc5-f0fb7ef5.pth) |
| DETR-R101    |    R-101   |  25                |  174.7 B |  55.7 M |   43.1   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r101-2c7b67e5.pth) |
| DETR-R101-DC5|    R-101   |  14                |  241.6 B |  55.7 M |   44.3   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/detr-r101-dc5-a2e86def.pth) |

## Anchor DETR
I evaluate AnchorDETR by loading official pretrained weight.

| Model               |  backbone  |  FPS<sup><br>3090  |  FLOPs   |  Params |    AP    |  Weight  |
|---------------------|------------|--------------------|----------|---------|----------|----------|
| Anchor-DETR-R50     |    R-50    |       37           |  97.0 B  |  30.7 M |   42.1   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r50_c5.pth) |
| Anchor-DETR-R50-DC5 |    R-50    |       20           |  154.0 B |  30.7 M |   44.2   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r50_dc5.pth) |
| Anchor-DETR-R101    |    R-101   |       21           |  176.5 B |  49.7 M |   43.5   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r101.pth) |
| Anchor-DETR-R101-DC5|    R-101   |       16           |  233.5 B |  49.7 M |   45.2   | [github](https://github.com/yjh0410/DeTR-LAB/releases/download/detr_weight/AnchorDETR_r101_dc5.pth) |



## Train
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

***However, limited by my GPU, I can't verify whether this repository can reproduce the performance of the official DETR.***

# Test
```Shell
python test.py -d coco \
               --cuda \
               -v detr_r50 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

# Evaluation
```Shell
python eval.py -d coco-val \
               --cuda \
               -v detr_r50 \
               --weight path/to/weight \
               --root path/to/dataset/
```

# Demo
I have provide some images in `data/demo/images/`, so you can run following command to run a demo:

```Shell
python demo.py --mode image \
               --path_to_img dataset/demo/images/ \
               -v detr_r50 \
               --cuda \
               --weight path/to/weight \
               --show
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_img dataset/demo/videos/video_name \
               -v detr_r50 \
               --cuda \
               --weight path/to/weight \
               --show
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               -v detr_r50 \
               --cuda \
               --weight path/to/weight \
               --show
```
