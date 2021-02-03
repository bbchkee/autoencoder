#!/bin/bash
#MSUB -l nodes=1:ppn=20
#MSUB -l walltime=259200
cd /project/fh2-project-tunkarex/xj1574/soft/denoiser
python trainer.py --signal cuted_signals.npy --noise cuted_noise.npy --min 100 --max 200 --epochs 100
exit 0

