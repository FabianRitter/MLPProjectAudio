#!/usr/local/bin/bash

for experiment in {1..9}
do
	echo Doing experiment number $experiment 
	python preprocess_data_new.py -p config/params.yaml -e $experiment
done
