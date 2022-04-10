rm -rf checkpoints
rm stdout.log

PYTHONPATH="." python tools/run_net.py --cfg /vision/u/naagnes/github/TimeSformer/configs/STIP/stip.yaml 
