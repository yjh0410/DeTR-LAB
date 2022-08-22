# Train
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v detr_r50 \
        --eval_epoch 10 \
        --no_warmup \
        --aux_loss \
        --use_nms \
        --pretrained weights/coco/detr_r50/detr-r50-e632da11.pth
        # --resume weights/coco/detr_r50/detr-r50-e632da11.pth
