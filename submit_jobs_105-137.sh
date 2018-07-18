for value in {105..138}
do
  RUN_NUM=value
  export RUN_NUM
  qsub monte_carlo_105-137.sh
done
