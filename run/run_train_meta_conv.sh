#!/bin/bash

rm -rf ../fl_train*.txt
rm -rf ../log
mkdir -p ../log/phase2/convection/sin_1

pwd

cd .. && bsub -K < run/run_train_meta_conv.bsub 
cd $HOMEDIR