#!/bin/bash

search_dir=/home/jans/PolyHok/results/nn
for entry in "$search_dir"/*
do 
  python3 testing-csv.py "$entry"
done
