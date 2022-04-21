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
source /mnt/disks/homography/stip_env/.venv/bin/activate
sh scripts/gcp_train.sh
```

## Tensorboard
```
tensorboard --logdir runs-stip/ --host 0.0.0.0
```

# TODOs:
- Freeze base encoder
- Make view specific encoders much bigger
- Scale to multiple GPUs
- Increase batch size from 1

## Extras

To kill a running python process, insert the PID here:
````
python -c "import os; import signal; os.kill(<pid>, signal.SIGTERM)"
````
