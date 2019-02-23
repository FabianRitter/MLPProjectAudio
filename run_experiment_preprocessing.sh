#!/usr/local/bin/bash

for experiment in {1..5}
do
	echo Doing experiment number for validation $experiment 
	python preprocess_data_new.py -p config/params_val.yaml -e $experiment
done

echo finished validation

for experiment in {1..15}
do
	echo Doing experiments for tests number $experiment
	python preprocess_data_new.py -p config/params_test -e $experiment
done

for experiment in {1..271}
do
	echo doing experiments for training number $experiment
	python preprocess_data_new.py -p config/params_train -e $experiment
done
