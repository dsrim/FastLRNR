#!/bin/bash

rm -rf ../fl_train*.txt
rm -rf ../log
mkdir -p ../log/phase2/convection/sin_1
rm -rf ../_plot

pwd

cd .. && bsub -K < run/run_test_conv.bsub 
cd $HOMEDIR
