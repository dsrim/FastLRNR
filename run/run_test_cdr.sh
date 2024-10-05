#!/bin/bash

rm -rf ../fl_train*.txt
rm -rf ../log
mkdir -p ../log/phase2/convection/sin_1
mkdir -p ../log/phase2/cdr/sin_1
rm -rf ../_plot

pwd
export RUNDIR=$PWD 

cd .. \ 
&& bsub -K < run/run_test_cdr.bsub \
&& bsub -K < run/data_up.bsub

cd $HOMEDIR
