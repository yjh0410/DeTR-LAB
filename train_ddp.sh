# 8 GPUs
python -m torch.distributed.run --nproc_per_node=8 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v detr_r18 \
                                                    --num_workers 4 \
                                                    --eval_epoch 10 \
                                                    --no_warmup \
                                                    --aux_loss \
