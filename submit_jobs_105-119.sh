#!/bin/bash

for value in {105..119}
do
  RUN_NUM=$value
  export RUN_NUM
  qsub -v RUN_NUM monte_carlo_105-119.sh
done
