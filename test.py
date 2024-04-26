import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Block(nn.Module):
	def __init__(self, embeddings, headcount, context_size):
		super().__init__()
		head_size = embeddings // headcount
		self.sa = MultiHeadAttention(headcount, head_size, embeddings, context_size, 0.2)
		self.ffwd = FeedForward(embeddings, 0.2)

		self.ln1 = nn.LayerNorm(embeddings)
		self.ln2 = nn.LayerNorm(embeddings)

	def forward(self, x):
		y = x.clone().detach()
		y =+ self.sa(self.ln1(x))
		y += self.ffwd(self.ln2(x))
		return y

class MultiHeadAttention(nn.Module):
	def __init__(self, headcount, headsize, embeddings, context_size, d_chance):
		super().__init__()
		self.heads= nn.ModuleList([Head(embeddings, headsize, context_size, d_chance) for _ in range(headcount)])
		self.proj = nn.Linear(headsize * headcount, embeddings)
		self.dropout = nn.Dropout(d_chance)

	def forward(self,x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		return self.dropout(self.proj(out))

class Head(nn.Module):
	def __init__(self, dims, head_sz, size, d_chance):
		super().__init__()
		self.key = nn.Linear(dims, head_size, bias=False)
		self.query = nn.Linear(dims, head_size, bias=False)
		self.value = nn.Linear(dims, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(size, size)))
		self.dropout = nn.Dropout(d_chance)


	def forward(self, x):
		B, T, C = x.shape
		key = self.key(x)
		query = self.query(x)


		weight = query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)
		weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		weight = self.dropout(F.softmax(weight, dim=-1))

		value = self.value(x)
		out = weight @ value
		return out

class FeedForward(nn.Module):
	def __init__(self, embeddings, d_chance):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(embeddings, 4*embeddings),
			nn.ReLU(),
			nn.Linear(4*embeddings, embeddings),
			nn.Dropout(d_chance))

	def forward(self, x):
		return self.net(x)

class Model(nn.Module):
	def __init__(self, context_size, batche_count, vocabulary_size, embed_dims, headcount):
		super().__init__()
		self.context_size = context_size
		self.batch_count = batche_count
		self.dims = embed_dims

		self.token_embedding_table = nn.Embedding(vocabulary_size, embed_dims)
		self.position_embedding_table = nn.Embedding(context_size, embed_dims)

		self.blocks = nn.Sequential(
			Block(embed_dims, headcount, context_size),
			Block(embed_dims, headcount, context_size),
			Block(embed_dims, headcount, context_size),
			nn.LayerNorm(embed_dims)
		)
		self.lm_head = nn.Linear(embed_dims, vocabulary_size)

	def get_batch(self, data):
		ran = torch.randint(len(data) - self.context_size, (self.batch_count,))
		batch_in = torch.stack([data[i:i+self.context_size] for i in ran])
		batch_out = torch.stack([data[i+1:i+self.context_size+1] for i in ran])
		batch_in, batch_out = batch_in.to(device), batch_out.to(device)
		return batch_in, batch_out

	def forward(self, idx, targets=None):
		ba, wi = idx.shape
		token_embed = self.token_embedding_table(idx) # (Batch x width) -> (Batch x width x dms)
		pos_emb = self.position_embedding_table(torch.arange(wi, device=device))
		tok_pos = token_embed + pos_emb
		tok_pos = self.blocks(tok_pos)
		logits = self.lm_head(tok_pos)

		if None == targets:
			loss = None
		else:
			# torch expects mD input for cross entropy to be formatted as (batches, channels, width)
			# where channels is the vocabulary size
			ba, wi, ch = logits.shape
			logits = logits.view(ba * wi, ch)
			targets = targets.view(ba*wi)
			loss = F.cross_entropy(logits, targets)
		return logits, loss

	def predict(self, idx, maximal_tokens):
		for _ in range(maximal_tokens):
			indx = idx[:, -self.batch_count:]
			logits, _ = self(indx)
			logits = logits[:, -1, :]
			distrib = F.softmax(logits, dim=-1)
			idx_next = torch.multinomial(distrib, num_samples=1)
			idx = torch.cat((idx, idx_next), dim=1)
		return idx

	@torch.no_grad()
	def estimate_loss(self, count, data):
		self.eval()
		losses = torch.zeros(count)
		for x in range(count):
			b_in, b_out = self.get_batch(data)
			logits, loss = m(b_in, b_out)
			losses[x] = loss.item()
		out = losses.mean()
		self.train()
		return out

if __name__ == '__main__':
	print('Beginning...\n\n')
	print('Running on',device)
	with open('input.txt', 'r', encoding='utf-8') as f:
		text = f.read()
	print(f'Read text [{len(text)}]')
	
	words = sorted(list(set(text)))
	sz = int(0.9*len(text))
	stoi = {wrd:i for i,wrd in enumerate(words)}
	itos = {i:wrd for i,wrd in enumerate(words)}

	wtov = lambda w: [stoi[wrd] for wrd in w]
	vtow = lambda v: ''.join([itos[vec] for vec in v])
	
	data = torch.tensor(wtov(text), dtype=torch.long)
	training = data[:sz]
	validation = data[sz:]

	batch_count = 4
	size = 128 #context
	head_size = 4
	dims = 16
	iters = 50
	learn = 3e-4

	print('Setup variables and lambdas.')

	m = Model(size, batch_count, len(training), dims, head_size)
	m = m.to(device)	

	optimizer = torch.optim.AdamW(m.parameters(), lr=learn)

	print('Setup model')
	print('\n\n')
	for steps in range(iters):
		print(f'{100*steps/iters}%')
		batch_in, batch_out = m.get_batch(training)
		_, loss = m(batch_in, batch_out)

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

	print("finished training.")
	con = torch.zeros((1,1), dtype=torch.long, device=device)
	pred = m.predict(con, maximal_tokens=500)[0].tolist()
	print(f'\n\nWith a loss of {m.estimate_loss(iters//10, training)} we have {vtow(pred)}')