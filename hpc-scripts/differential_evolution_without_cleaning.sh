#!/bin/bash
#PBS -N DifferentialEvolutionRun130
#PBS -j oe
#PBS -l walltime=80:00:00
#PBS -l nodes=1:ppn=1
#PBS -l pmem=5gb
#PBS -m a
#PBS -M chchen@davidson.edu

#This script submits PBS job commands to the hpc, using configurations and paths to all Ar46 files

RUN_NUM=130
RUN_NUM_PADDED=`printf "%04d" ${RUN_NUM}`
echo "$RUN_NUM_PADDED"

CONFIG_FILE="/mnt/home/mkuchera/ATTPC/ar46_reqfiles/config_e15503b_p.yml"
DATA_ROOT="/mnt/research/attpc/analysis/e15503b/cleanh5"
OUTPUT_DIR="/mnt/research/attpc/data/e15503b/DEtest/without_cleaning"

INPUT_FILE=${DATA_ROOT}/clean_run_${RUN_NUM_PADDED}.h5
OUTPUT_FILE=${OUTPUT_DIR}/run_${RUN_NUM_PADDED}.h5

TEMP_DIR=/tmp/${USER}
INPUT_FILE_TEMP=${TEMP_DIR}/clean_run_${RUN_NUM_PADDED}.h5
OUTPUT_FILE_TEMP=${TEMP_DIR}/run_${RUN_NUM_PADDED}.h5

if [ -e $TEMP_DIR ]; then
   echo "temp dir exists"
else
   mkdir ${TEMP_DIR}
fi

if [ -e $INPUT_FILE ]; then
   cd $OUTPUT_DIR
   cp ${INPUT_FILE} ${TEMP_DIR}/.

   /mnt/home/mkuchera/external/Python-3.6.1/python ${HOME}/ATTPC/pytpc/bin/differential_evolution_without_cleaning.py ${CONFIG_FILE} ${INPUT_FILE_TEMP} ${OUTPUT_FILE_TEMP}

   cp ${OUTPUT_FILE_TEMP} ${OUTPUT_FILE}
   rm ${INPUT_FILE_TEMP}
   rm ${OUTPUT_FILE_TEMP}

   echo "Data fitted successfully"
else
   echo "File does not exist"
fi
