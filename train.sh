#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=train_lr0008_0001_00001.log

pwd
export PYTHONPATH=$(pwd):$PYTHONPATH

abort() { >&2 printf '█%.0s' {1..40}; (>&2 printf "\n[ERROR] $(basename $0) has exited early\n"); exit 1; }
scriptdirpath=$(cd -P -- "$(dirname -- "$0")" && pwd -P);
thisscript=$scriptdirpath/$(basename $0)
IFS=$'\n\t'; set -eo pipefail;
trap 'abort' 0; set -u; # Loads abort() trap defined in line 2
# set -u exits when detects unset variables
# if [ -z "$*" ]; then echo "No args"; exit 1; fi

# _______________ END OF SAFE CRASH SETTINGS _______________

# Requirements: Conda env to be setup (environment.yml)
#               and ... files in the same directory as this.

# Args: Python scripts in the order you wish to run.

# Purpose: Executes python scripts in the IBME cluster's GPU node.

# _______________________ ACTUAL CODE ______________________

# cd into the scriptdirpath so that relative paths work
#pushd "${scriptdirpath}" > /dev/null

CONDA_ENV="vqloc"
ml cuda

# conda env stuff
. "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
  echo "activating ${CONDA_ENV} env"
  set +u; conda activate "${CONDA_ENV}"; set -u
else
  echo "current conda env: '${CONDA_DEFAULT_ENV}' = '${CONDA_ENV}'"
fi

export WANDB_API_KEY=1234
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 9999 --nproc_per_node=8 \
# CUDA_VISIBLE_DEVICES=0 python ./train_anchor.py --cfg ./config/pulse_train.yaml
for f in "./config/pulse_train"*; do
  echo "python ./train_anchor.py --cfg $f"
  python ./train_anchor.py --cfg $f

done
# conda deactivate

#popd > /dev/null
# srun -G1 --nodelist=node05 --pty bash

trap : 0
(>&2 echo "✔")
exit 0
