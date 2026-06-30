#!/bin/bash

# et="loo/"
# A=($et"too baa" 
#   $et"bar marulaa" 
#   "loo" 
#   "too" 
#   "moo")
#
# for x in "${A[@]}"
# do
#   echo "$x"
# done
#
cd ~/PolyHok/


NB=(
  "mix run experiments/nbody/nbody_bench.exs --size=102400 --type=integer --order=0"
  "mix run experiments/nbody/nbody_bench.exs --size=102400 --type=integer --order=1"
  "mix run experiments/nbody/nbody_bench.exs --size=102400 --type=float --order=0"
  "mix run experiments/nbody/nbody_bench.exs --size=102400 --type=float --order=1"
  "mix run experiments/nbody/nbody_bench.exs --size=102400 --type=double --order=0"
  "mix run experiments/nbody/nbody_bench.exs --size=102400 --type=double --order=1"
  "mix run experiments/nbody/nbody_bench.exs --size=204800 --type=integer --order=0"
  "mix run experiments/nbody/nbody_bench.exs --size=204800 --type=integer --order=1"
  "mix run experiments/nbody/nbody_bench.exs --size=204800 --type=float --order=0"
  "mix run experiments/nbody/nbody_bench.exs --size=204800 --type=float --order=1"
  "mix run experiments/nbody/nbody_bench.exs --size=204800 --type=double --order=0"
  "mix run experiments/nbody/nbody_bench.exs --size=204800 --type=double --order=1"
  "mix run experiments/nbody/nbody_bench.exs --size=409600 --type=integer --order=0"
  "mix run experiments/nbody/nbody_bench.exs --size=409600 --type=integer --order=1"
  "mix run experiments/nbody/nbody_bench.exs --size=409600 --type=float --order=0"
  "mix run experiments/nbody/nbody_bench.exs --size=409600 --type=float --order=1"
  "mix run experiments/nbody/nbody_bench.exs --size=409600 --type=double --order=0"
  "mix run experiments/nbody/nbody_bench.exs --size=409600 --type=double --order=1"
  )

NI=(
  "mix run experiments/numeric_integration/ni_bench.exs --fun=v1 --order=0"
  "mix run experiments/numeric_integration/ni_bench.exs --fun=v1 --order=1"
  "mix run experiments/numeric_integration/ni_bench.exs --fun=v2 --order=0"
  "mix run experiments/numeric_integration/ni_bench.exs --fun=v2 --order=1"
  "mix run experiments/numeric_integration/ni_bench.exs --fun=v3 --order=0"
  "mix run experiments/numeric_integration/ni_bench.exs --fun=v3 --order=1"
  )

NN=(
  "mix run experiments/nearest_neighbor/nn_bench.exs --file=small.csv --lat=25.0 --lng=290.0 --order=0"
  "mix run experiments/nearest_neighbor/nn_bench.exs --file=small.csv --lat=25.0 --lng=290.0 --order=1"
  "mix run experiments/nearest_neighbor/nn_bench.exs --file=medium.csv --lat=25.0 --lng=290.0 --order=0"
  "mix run experiments/nearest_neighbor/nn_bench.exs --file=medium.csv --lat=25.0 --lng=290.0 --order=1"
  "mix run experiments/nearest_neighbor/nn_bench.exs --file=large.csv --lat=25.0 --lng=290.0 --order=0"
  "mix run experiments/nearest_neighbor/nn_bench.exs --file=large.csv --lat=25.0 --lng=290.0 --order=1"
  )
PP=(
  "mix run experiments/portfolio_pricing/pp_bench.exs --size=10240000 --order=0"
  "mix run experiments/portfolio_pricing/pp_bench.exs --size=10240000 --order=1"
  "mix run experiments/portfolio_pricing/pp_bench.exs --size=20480000 --order=0"
  "mix run experiments/portfolio_pricing/pp_bench.exs --size=20480000 --order=1"
  "mix run experiments/portfolio_pricing/pp_bench.exs --size=30720000 --order=0"
  "mix run experiments/portfolio_pricing/pp_bench.exs --size=30720000 --order=1"
  )
SAXPY=(
  "mix run experiments/axpy/axpy_bench.exs --size=10240000 --type=float --order=0"
  "mix run experiments/axpy/axpy_bench.exs --size=10240000 --type=integer --order=0"
  "mix run experiments/axpy/axpy_bench.exs --size=10240000 --type=double --order=0"
  "mix run experiments/axpy/axpy_bench.exs --size=10240000 --type=float --order=1"
  "mix run experiments/axpy/axpy_bench.exs --size=10240000 --type=integer --order=1"
  "mix run experiments/axpy/axpy_bench.exs --size=10240000 --type=double --order=1"
  "mix run experiments/axpy/axpy_bench.exs --size=20480000 --type=float --order=0"
  "mix run experiments/axpy/axpy_bench.exs --size=20480000 --type=integer --order=0"
  "mix run experiments/axpy/axpy_bench.exs --size=20480000 --type=double --order=0"
  "mix run experiments/axpy/axpy_bench.exs --size=20480000 --type=float --order=1"
  "mix run experiments/axpy/axpy_bench.exs --size=20480000 --type=integer --order=1"
  "mix run experiments/axpy/axpy_bench.exs --size=20480000 --type=double --order=1"
  "mix run experiments/axpy/axpy_bench.exs --size=30720000 --type=float --order=0"
  "mix run experiments/axpy/axpy_bench.exs --size=30720000 --type=integer --order=0"
  "mix run experiments/axpy/axpy_bench.exs --size=30720000 --type=double --order=0"
  "mix run experiments/axpy/axpy_bench.exs --size=30720000 --type=float --order=1"
  "mix run experiments/axpy/axpy_bench.exs --size=30720000 --type=integer --order=1"
  "mix run experiments/axpy/axpy_bench.exs --size=30720000 --type=double --order=1"
  )

DP=(
  "mix run experiments/dot_product/dp_bench.exs --size=10240000 --type=float --order=0"
  "mix run experiments/dot_product/dp_bench.exs --size=10240000 --type=integer --order=0"
  "mix run experiments/dot_product/dp_bench.exs --size=10240000 --type=double --order=0"
  "mix run experiments/dot_product/dp_bench.exs --size=10240000 --type=float --order=1"
  "mix run experiments/dot_product/dp_bench.exs --size=10240000 --type=integer --order=1"
  "mix run experiments/dot_product/dp_bench.exs --size=10240000 --type=double --order=1"
  "mix run experiments/dot_product/dp_bench.exs --size=20480000 --type=float --order=0"
  "mix run experiments/dot_product/dp_bench.exs --size=20480000 --type=integer --order=0"
  "mix run experiments/dot_product/dp_bench.exs --size=20480000 --type=double --order=0"
  "mix run experiments/dot_product/dp_bench.exs --size=20480000 --type=float --order=1"
  "mix run experiments/dot_product/dp_bench.exs --size=20480000 --type=integer --order=1"
  "mix run experiments/dot_product/dp_bench.exs --size=20480000 --type=double --order=1"
  "mix run experiments/dot_product/dp_bench.exs --size=30720000 --type=float --order=0"
  "mix run experiments/dot_product/dp_bench.exs --size=30720000 --type=integer --order=0"
  "mix run experiments/dot_product/dp_bench.exs --size=30720000 --type=double --order=0"
  "mix run experiments/dot_product/dp_bench.exs --size=30720000 --type=float --order=1"
  "mix run experiments/dot_product/dp_bench.exs --size=30720000 --type=integer --order=1"
  "mix run experiments/dot_product/dp_bench.exs --size=30720000 --type=double --order=1"
  )
for run in "${NN[@]}"
do 
  $run
done
