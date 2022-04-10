# 3D Aware Unsupervised Representation Learning

## Training

```
python tools/run_net.py \
  --cfg configs/STIP/stip.yaml \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

# TODOs:
- Fix loss function
- Change transformation to absolute minimum.
- Use a pretrained model.
- Setup tensorboard training for loss logging.
- Train with batch size of 1.
- Load pretrained model.
- FIX TRASNFORMATIONS
- Need a label number to get the correct homography matrix.