#!/bin/bash

prog='./axpy'

for ((i = 0; i < 10; i++)); do
  printf "%d " "$i"; $prog
done
