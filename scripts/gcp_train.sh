rm -rf checkpoints
rm stdout.log

PYTHONPATH="." python tools/run_net.py --cfg /home/nico/TimeSformer/configs/STIP/stip_gcp.yaml 
