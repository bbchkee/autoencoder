#!/bin/bash
#MSUB -l nodes=1:ppn=48:visu
#MSUB -l walltime=259200
#export PYTHONPATH=/project/fh2-project-tunkarex/xj1574/soft/python/lib64/python2.7/site-packages
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/all/devel/cuda-8.0/lib64:/project/fh2-project-tunkarex/xj1574/soft/cuda-lib/cuda/lib64:/software/all/devel/cuda-8.0/lib64/stubs
#nvcc --version
#hostname
cd /project/fh2-project-tunkarex/xj1574/soft/denoiser
python trainer.py --signal cuted_signals.npy --noise cuted_noise.npy --min 100 --max 200 --epochs 500
exit 0

