#!/usr/bin/bash -l
#SBATCH --job-name=MYVERSION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-a10 #zen3
#SBATCH --cpus-per-task=1
#SBATCH --mem=99G
#SBATCH --time=5-00:00:00
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
#PROCESSES_NUM=16
#MINLEN=3000
EPOCH_NUM=15 # Default 15
METRIC="l2"
SUFFIX="_test1"

# Define Samples
SAMPLES=( "soil_MFD00392" )
#( "wwtp_IL-01" "wwtp_US2-26" "sediment_MFD10007" "soil_MFD00392" )
#( "ZymoFecal" ) 
#( "PaPr00000116MP" ) #( "sediment_MFD10007" ) #( "wwtp_US2-26" ) #( "soil_MFD00392" ) #( "ZymoFecal" ) #( "wwtp_IL-01" ) # wwtp_IL-01  wwtp_US2-26  ( "ZymoFecal" )
SAMPLES_NUM=${#SAMPLES[@]}

## Load the required modules
SNAKEMAKE=/home/cs.aau.dk/zs74qz/.conda/envs/snakemake/bin/snakemake
## Activate the conda
conda activate ${CONDA_ENV_DIR}
## Install the current SemiBin2 version
#${CONDA_ENV_DIR}/bin/pip install ${BASE_DIR}
export PYTHONPATH=${BASE_DIR}

# Run
for (( i=0; i<$SAMPLES_NUM; i++ )); do

   SAMPLE_NAME="${SAMPLES[$i]}"
   echo ${SAMPLE_NAME}

   # Input files
   CONTIG_FILE=${DATA_DIR}/${SAMPLE_NAME}/contigs.fasta
   BAM_FILE=${DATA_DIR}/${SAMPLE_NAME}/1-NP.bam
   # Output files
   OUTPUT_DIR=${BASE_DIR}/outputs/${SAMPLE_NAME}${SUFFIX}
   OUTPUT_FILE=${BASE_DIR}/outputs/${SAMPLE_NAME}${SUFFIX}/test.model

   CMD="${PYTHON_PATH} ${BASE_DIR}/src/model.py"
   CMD="${CMD} --input ${CONTIG_FILE} --output ${OUTPUT_FILE}"
   CMD="${CMD} --neg_sample_per_pos 500 --epoch 15 --max_seq_num 0"
   CMD="${CMD} --loss_name bern_hinge_squared --out_dim 100"

   ${CMD}

done

