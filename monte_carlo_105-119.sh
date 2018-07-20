#!/bin/bash
#PBS -N MonteCarloTest105-119
#PBS -j oe
#PBS -o /mnt/home/mkuchera/ATTPC/ar40_jobs/MonteCarlo/MC_output.o
#PBS -l walltime=20:00:00
#PBS -l nodes=1:ppn=1
#PBS -l pmem=5gb
#PBS -m a
#PBS -M chchen@davidson.edu
echo $RUN_NUM
RUN_NUM_PADDED=`printf "%04d" ${RUN_NUM}`
echo "$RUN_NUM_PADDED"

CONFIG_FILE="/mnt/home/mkuchera/ATTPC/ar40_reqfiles/config_e15503a_runs_105-137.yml"
DATA_ROOT="/mnt/research/attpc/data/e15503a/hdf5_cleaned"
OUTPUT_DIR="/mnt/research/attpc/data/e15503a/mc_test"

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

   /mnt/home/mkuchera/external/Python-3.6.1/python ${HOME}/ATTPC/pytpc/bin/Monte_carlo.py ${CONFIG_FILE} ${INPUT_FILE_TEMP} ${OUTPUT_FILE_TEMP}

   cp ${OUTPUT_FILE_TEMP} ${OUTPUT_FILE}
   rm ${INPUT_FILE_TEMP}
   rm ${OUTPUT_FILE_TEMP}

   echo "Data fitted successfully"
else
   echo "File does not exist"
fi
