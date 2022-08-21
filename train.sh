# Train
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v detr_r50 \
        --eval_epoch 10 \
        --aux_loss \
        --use_nms \
        --resume weights/coco/detr_r50/detr-r50-e632da11.pth
