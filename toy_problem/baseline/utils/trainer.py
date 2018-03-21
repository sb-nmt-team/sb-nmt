import torch
import numpy as np
import tqdm
from torch.autograd import Variable

from metrics.bleu import bleu_from_lines
from utils.hparams import HParams
# fix it


class Trainer:
	def __init__(self, model, batch_sampler, hps, training_hps):
		self.hps = hps
		self.training_hps = training_hps

		self.model = model
		self.batch_sampler = batch_sampler
		self.losses = []

		# may be create multiple metrics, not only bleu
		self.bleu = []

	def reset(self):
		pass

	def for_translation(self, x, x_mask):
		if not self.training_hps.use_cuda:
			x = Variable(torch.from_numpy(x.astype(np.int64))).contiguous()
			x_mask = Variable(torch.from_numpy(x_mask.astype(np.float32))).contiguous()
		else:
			x = Variable(torch.from_numpy(x.astype(np.int64))).contiguous().cuda()
			x_mask = Variable(torch.from_numpy(x_mask.astype(np.float32))).contiguous().cuda()

		return x, x_mask

	def run_translation(self, src, model, test_data, batch_size):
		result = []
		for pos in range(0, test_data.shape[0], batch_size):
			batch, mask = self.for_translation(*src.convert_batch(test_data[pos:pos + batch_size]))
			translated = model.translate(batch, mask)
			result.extend(translated)

		real_result = []
		for sent in result:
			sent = sent.split(" ")[1:-1]
			real_result.append(" ".join(sent))
		return real_result

	def validate(self):
		test_data = self.batch_sampler.dev[0]
		translation = self.run_translation(self.batch_sampler.get_src(), self.model, test_data,
		                                   self.training_hps.batch_size)
		real_translation = [' '.join(x) for x in self.batch_sampler.dev[1]]
		return bleu_from_lines(real_translation, translation)

	def train(self):
		self.model.train()

		# todo multiple optimizers
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_hps.starting_learning_rate)

		for epoch_id in range(self.training_hps.n_epochs):

			for batch_id, ((input, input_mask), (output, output_mask)) in \
				tqdm.tqdm(enumerate(self.batch_sampler), total=len(self.batch_sampler)):
				if self.training_hps.use_cuda:
					input = input.cuda()
					input_mask = input_mask.cuda()
					output = output.cuda()
					output_mask = output_mask.cuda()

				loss = self.model(input, input_mask, output, output_mask)
				optimizer.zero_grad()

				loss.backward()
				torch.nn.utils.clip_grad_norm(self.model.parameters(), self.training_hps.clip)
				optimizer.step()

				# todo create hooks
				# it's really doubtfull to hold all of them
				if self.training_hps.use_cuda:
					self.losses.append(loss.cpu().data[0])
				else:
					self.losses.append(loss.data[0])
				print(loss.cpu().data[0])
				# if (batch_id * batch_sampler.batch_size) % 1000 == 0:
				# 	display.clear_output(wait=True)
				# 	print("Last 10 loses mean", np.mean(losses[-10:]))
				# 	plt.plot(losses)
				# 	plt.show()
			self.model.eval()
			self.bleu.append(self.validate())

			print("Bleu: ", self.bleu[-1])
			self.model.train()

			# todo redo the saving
			# torch.save(self.model.state_dict(), "last_state.ckpt")
			# gc.collect()
			if self.training_hps.use_cuda:
				torch.cuda.empty_cache()

	def get_metrics(self):
		# todo return metrics and not this stuff
		return self.losses, self.bleu

	@staticmethod
	def get_default_hparams():
		return HParams(
			use_cuda=False,
			max_length=15,
			batch_size=128,
			n_epochs=40,
			clip=0.25,
			starting_learning_rate=1e-3, # todo
			learning_rate_strategy="constant_decay", # todo
			optimizer="Adam" # todo
		)