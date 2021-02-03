#!/bin/bash
#MSUB -l nodes=1:ppn=20
#MSUB -l walltime=259200
export PYTHONPATH=$PYTHONPATH:/home/soft/python
export CUDA_VISIBLE_DEVICES=0
python /home/soft/denoiser/trainer.py --signal /home/soft/denoiser/cuted_signals.npy --noise /home/soft/denoiser/cuted_noise.npy --min 100 --max 200 --epochs 100
exit 0

