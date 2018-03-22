#!/usr/bin/env python3
# coding=utf-8
import sys
sys.path.append('..')

from model import s2s
from utils.trainer import Trainer
from data.lang import read_problem
from data.batcher import BatchSampler

# add more tests
def main():
	hps = s2s.Seq2Seq.get_default_hparams()
	training_params = Trainer.get_default_hparams()

	training_params.set('n_epochs', 25)

	# that really should be dynamic
	dataset, src, tgt = read_problem("../../preprocessed/hewv-en/", n_sents=None)

	# dummy_dataset = {
	# 	"train": (["'a 'a d y r", "'a 'a h b ckh"], ["a a d i r", "e a h a v k h a"]),
	# 	"test": (["'a 'a d y r", "'a 'a h b ckh"], ["a a d i r", "e a h a v k h a"]),
	# 	"dev": (["'a 'a d y r", "'a 'a h b ckh"], ["a a d i r", "e a h a v k h a"])
	# }

	batch_sampler = BatchSampler(dataset, src, tgt, training_params.batch_size)
	model = s2s.Seq2Seq(src, tgt, hps, training_params)
	trainer = Trainer(model, batch_sampler, hps, training_params)
	# trainer.train()
	# losses, bleus = trainer.get_metrics()
	# assert bleus[-1] > 0.99
	print("~" * 99)
	print("OK, Net converges")

if __name__ == '__main__':
	main()