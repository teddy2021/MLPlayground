from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import re
import test
from functools import reduce


flatmap = lambda fcn, lst: reduce(lambda a, b: a + b, map(fcn, lst))

ds = load_dataset("garage-bAInd/Open-Platypus", split='train')
print(ds.features)
ds.set_format(type='torch', columns=['input','output', 'instruction','data_source'])
ds = ds.remove_columns(column_names=['input', 'data_source'])
prompts = ds["instruction"]
prompt_tokens = list(map(lambda x: x.split(), prompts))
print(prompt_tokens[0])
responses = ds['output']
response_tokens = list(map(lambda x: x.split(), responses))
sentences = prompts + responses
tokens = []

for x in sentences:
	tokens += re.split(r"([^\w\t\n\r\f\v])", x)
tokens = list(set(tokens))
toid = {wrd:i for i, wrd in enumerate(tokens)}
tostr = {i:wrd for i, wrd in enumerate(tokens)}
print(toid['A'])
prompt_ids = torch.tensor(flatmap(test.encode,prompt_tokens), dtype=torch.long)
response_ids = torch.tensor(flatmap(test.encode,response_tokens), dtype=torch.long)

print(prompt_ids[:10])
print(response_ids[:10])