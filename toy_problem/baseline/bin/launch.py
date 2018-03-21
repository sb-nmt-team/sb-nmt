#!/usr/bin/env python3
# coding=utf-8
import argparse
import json
import sys
sys.path.append('..')

from model import s2s
from utils.trainer import Trainer
from data.lang import read_problem
from data.batcher import BatchSampler

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--params')
	parser.add_argument('--training_params')
	args = parser.parse_args()

	# todo logdir
	# todo load from logdir if possible, load params
	hps = s2s.Seq2Seq.get_default_hparams()
	if args.params:
		with open(args.params, encoding='utf-8') as fin:
			hps = hps.parse_dict(json.load(fin))

	training_params = Trainer.get_default_hparams()
	if args.training_params:
		with open(args.training_params, encoding='utf-8') as fin:
			training_params = training_params.parse_dict(json.load(fin))

	# that really should be dynamic
	dataset, src, tgt = read_problem("../../preprocessed/he-en/", n_sents=None)

	dummy_dataset = {
		"train": (["'a 'a d y r", "'a 'a h b ckh"], ["a a d i r", "e a h a v k h a"]),
		"test": (["'a 'a d y r", "'a 'a h b ckh"], ["a a d i r", "e a h a v k h a"]),
		"dev": (["'a 'a d y r", "'a 'a h b ckh"], ["a a d i r", "e a h a v k h a"])
	}
	batch_sampler = BatchSampler(dummy_dataset, src, tgt, training_params.batch_size)

	model = s2s.Seq2Seq(src, tgt, hps, training_params)

	trainer = Trainer(model, batch_sampler, hps, training_params)
	trainer.train()

if __name__ == '__main__':
	main()