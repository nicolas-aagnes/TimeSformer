# 3D Aware Unsupervised Representation Learning

## Installation
```
pip install --upgrade pip
pip install setuptools==59.5.0
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install simplejson einops timm psutil scikit-learn opencv-python tensorboard av matplotlib
pip install 'git+https://github.com/facebookresearch/fvcore'
```

## Activate Environment
```
git push --set-upstream origin pairwise_loss
```

## Training
```
sh scripts/gcp_stip.sh
```

# TODOs:
- Setup tensorboard training for loss logging.