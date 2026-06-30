#!/bin/bash

PROG="./axpy"

for ((i = 0; i < 10; i++)); do  
  $(PROG)
done

echo "hello"
