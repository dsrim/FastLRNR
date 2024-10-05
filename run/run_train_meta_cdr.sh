#!/bin/bash

rm -rf ../fl_train*.txt
rm -rf ../log
mkdir -p ../log/phase2/cdr/sin_1

pwd

cd .. && bsub -K < run/run_train_meta_cdr.bsub 
cd $HOMEDIR