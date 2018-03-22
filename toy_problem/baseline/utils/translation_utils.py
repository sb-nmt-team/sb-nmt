import torch
import numpy as np
import tqdm
from torch.autograd import Variable



def for_translation(x, x_mask, params):
	if not params.use_cuda:
		x = Variable(torch.from_numpy(x.astype(np.int64))).contiguous()
		x_mask = Variable(torch.from_numpy(x_mask.astype(np.float32))).contiguous()
	else:
		x = Variable(torch.from_numpy(x.astype(np.int64))).contiguous().cuda()
		x_mask = Variable(torch.from_numpy(x_mask.astype(np.float32))).contiguous().cuda()

	return x, x_mask

def run_translation(src, model, test_data, params):
	batch_size = params.batch_size
	result = []
	for pos in range(0, test_data.shape[0], batch_size):
		x, x_mask = src.convert_batch(test_data[pos:pos + batch_size])
		batch, mask = for_translation(x, x_mask, params)
		translated = model.translate(batch, mask)
		result.extend(translated)

	real_result = []
	for sent in result:
		sent = sent.split(" ")[1:-1]
		real_result.append(" ".join(sent))
	return real_result