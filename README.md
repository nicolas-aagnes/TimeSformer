# 3D Aware Unsupervised Representation Learning

## Installation
```
pip install --upgrade pip
pip install setuptools==59.5.0
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install simplejson einops timm psutil scikit-learn opencv-python tensorboard av matplotlib
pip install 'git+https://github.com/facebookresearch/fvcore'
sudo apt-get install libbz2-dev/mnt/disks/homography/stip_env/.venv/bin/python -m pip install -U black
source /mnt/disks/homography/stip_env/.venv/bin/activate
```

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