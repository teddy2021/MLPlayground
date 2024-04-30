from datasets import load_dataset
from torch.utils.data import DataLoader
import re
import test
import torch
from torch.nn import Functional as F

class DPO:
	def __init__(self, model, beta=0.05):
		self.beta = beta
		self.frozen = model # the frozen model for evaluation 
		self.active = model # the model to be tuned

	def get_loss(self, given):
		frozen_acc, frozen_rej = self.frozen(given)
		active_acc, active_rej = self.active(given)

		frozen_acc = F.log_softmax(frozen_acc,1)
		frozen_rej = F.log_softmax(frozen_rej,1)
		active_acc = F.log_softmax(active_acc,1)
		active_rej = F.log_softmax(active_rej,1)

		frozen_ratio = frozen_acc - frozen_rej
		active_ratio = active_acc - active_rej




if __name__ == "__main__":
#	ds = load_dataset("garage-bAInd/Open-Platypus", split='train')
#	print(ds.features)
#	ds.set_format(type='torch', columns=['input','output', 'instruction','data_source'])
#	ds = ds.remove_columns(column_names=['input', 'data_source'])
#	prompts = ds["instruction"]
#	prompt_tokens = list(map(lambda x: re.split(r"([^\w\t\n\r\f\v])", x), prompts))
#	print(prompt_tokens[0])
#	responses = ds['output']
#	response_tokens = list(map(lambda x: re.split(r"([^\w\t\n\r\f\v])", x), responses))
#	sentences = prompts + responses
#	tokens = []
#
#	for x in sentences:
#		val = re.split(r"([^\w\t\n\r\f\v])", x)
#		tokens = tokens + val
#	tokens = list(set(tokens))
#	toid = {wrd:i for i, wrd in enumerate(tokens)}
#	tostr = {i:wrd for i, wrd in enumerate(tokens)}
#	prompt_ids = [] 
#	for prompt in prompt_tokens:
#		prompt_ids.append(test.encode(prompt, toid))
#	prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
#	response_ids = []
#	for response in response_tokens:
#		response_ids.append(test.encode(response, toid))
#	response_ids = torch.tensor(response_ids, dtype=torch.long)
#	print(prompt_ids[:10])
#	print(response_ids[:10])

