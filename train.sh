# Train
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v detr_r50 \
        --eval_epoch 10 \
        # --resume weights/coco/detr_r50/detr_r50.pth
