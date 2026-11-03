#!/usr/bin/bash -l
#SBATCH --job-name=CHECKM2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-a10 #zen3
#SBATCH --cpus-per-task=1
#SBATCH --mem=99G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL #--NONE #--ALL
#SBATCH --mail-user=abce@cs.aau.dk
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Exit on first error and if any variables are unset
set -eu

# Define global variables
BASE_DIR=/home/cs.aau.dk/zs74qz/workspace/newrepo
CONDA_ENV_DIR=/home/cs.aau.dk/zs74qz/.conda/envs/newrepo
PYTHON_PATH=${CONDA_ENV_DIR}/bin/python
#SEMIBIN2=${CONDA_ENV_DIR}/bin/SemiBin2
CHECKM2DB=/databases/checkm2/CheckM2_database/uniref100.KO.1.dmnd
CHECKM2=/home/cs.aau.dk/zs74qz/.conda/envs/checkm2_v1.1.0/bin/checkm2
DATA_DIR=/projects/dark_science/cs/samples
PROCESSES_NUM=16
SUFFIX="_test1"

# Define Samples
SAMPLES=( "ZymoFecal" )
SAMPLES_NUM=${#SAMPLES[@]}


## Load the required modules
SNAKEMAKE=/home/cs.aau.dk/zs74qz/.conda/envs/snakemake/bin/snakemake
## Activate the conda
conda activate ${CONDA_ENV_DIR}
## Install the current SemiBin2 version
export PYTHONPATH=${BASE_DIR}

# Run
for (( i=0; i<$SAMPLES_NUM; i++ )); do

   SAMPLE_NAME="${SAMPLES[$i]}"
   echo ${SAMPLE_NAME}

   OUTPUT_DIR=${BASE_DIR}/outputs/${SAMPLE_NAME}${SUFFIX}_eval

   # Run CheckM2
   conda activate checkm2_v1.1.0
   CMD="${CHECKM2} predict -i ${OUTPUT_DIR}/bins/* -o ${OUTPUT_DIR}/checkm2 -x .fa -t ${PROCESSES_NUM} --force"
   $CMD

done

