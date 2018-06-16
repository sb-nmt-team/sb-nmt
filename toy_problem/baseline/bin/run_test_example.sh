#!/bin/bash
python test.py --training_params ../../launch_dir/trainer_params.json --params ../../launch_dir/params.json --dataset ../../preprocessed/he-en/ --model_state ../../trained_models/experiment_on_2018-04-12_08_03_16/last_state.ckpt  < ../../preprocessed/he-en/src.test.txt > result.txt


