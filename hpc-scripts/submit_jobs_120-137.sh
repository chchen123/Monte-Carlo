#!/bin/bash

for value in {120..125}
do
  RUN_NUM=$value
  export RUN_NUM
  qsub -v RUN_NUM monte_carlo_120-137.sh
done
