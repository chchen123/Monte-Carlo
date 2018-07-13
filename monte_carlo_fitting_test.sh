
#!/bin/bash

RUN_NUM=137
RUN_NUM_PADDED=`printf "%04d" ${PBS_ARRAYID}`

CONFIG_FILE="/mnt/home/mkuchera/ATTPC/ar40_reqfiles/config_e15503a_runs_105-137.yml"

DATA_ROOT="/mnt/research/attpc/data/e15503a/hdf5_cleaned/"
OUTPUT_DIR="/mnt/research/attpc/data/e15503a/mc_test/"

INPUT_FILE=${DATA_ROOT}/clean_run_137.h5
OUTPUT_FILE=${OUTPUT_DIR}/run_0137.h5

TEMP_DIR=/tmp/${USER}
INPUT_FILE_TEMP=${TEMP_DIR}/clean_run_137.h5
OUTPUT_FILE_TEMP=${TEMP_DIR}/run_0137.h5


if [ -e $TEMP_DIR ]; then
   echo "temp dir exists"
else
   mkdir ${TEMP_DIR}
fi


if [ -e $INPUT_FILE ]; then
   cd $OUTPUT_DIR
   cp ${INPUT_FILE} ${TEMP_DIR}/.

   /mnt/home/mkuchera/external/Python-3.6.1/python ${HOME}/ATTPC/pytpc/bin/pseudocode.py ${CONFIG_FILE} ${INPUT_FILE_TEMP} ${OUTPUT_FILE_TEMP}

   cp ${OUTPUT_FILE_TEMP} ${OUTPUT_FILE}
   rm ${INPUT_FILE_TEMP}
   rm ${OUTPUT_FILE_TEMP}

   echo "File succesfully cleaned"
else
   echo "File does not exist"
fi
