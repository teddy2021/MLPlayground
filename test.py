import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt

def encode(wordset, mapping):
	out = []
	for word in wordset:
		encoding = mapping[word]
		out.append(encoding)
	return out

def decode(encoding, mapping):
	out = []
	for num in encoding:
		word = mapping[num]
		out.append(word)
	return out


class Head(nn.Module):
	def __init__(self, dims, head_sz, size, d_chance=0.2):
		super().__init__()
		self.key = nn.Linear(dims, head_sz, bias=False)
		self.query = nn.Linear(dims, head_sz, bias=False)
		self.value = nn.Linear(dims, head_sz, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(size, size)))
		self.dropout = nn.Dropout(d_chance)


	def forward(self, x):
		B, T, C = x.shape
		key = self.key(x)
		query = self.query(x)
		value = self.value(x)

		#weight = query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)
		#weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		#weight = self.dropout(F.softmax(weight, dim=-1))

		
		out = F.scaled_dot_product_attention(query, key, value) #weight @ value
		
		return out

class MultiHeadAttention(nn.Module):
	def __init__(self, headcount, headsize, embeddings, context_size, d_chance=0.2):
		super().__init__()
		self.heads= nn.ModuleList([Head(embeddings, headsize, context_size, d_chance) for _ in range(headcount)])
		self.proj = nn.Linear(headsize * headcount, embeddings)
		self.dropout = nn.Dropout(d_chance)

	def forward(self,x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = checkpoint.checkpoint(self.proj, out, use_reentrant=False)
		out = self.dropout(out)
		return out


class Block(nn.Module):
	def __init__(self, embeddings, headcount, context_size):
		super().__init__()
		head_size = embeddings // headcount
		self.sa = MultiHeadAttention(headcount, head_size, embeddings, context_size, 0.2)
		self.ffwd = FeedForward(embeddings, 0.2)

		self.ln1 = nn.LayerNorm(embeddings)
		self.ln2 = nn.LayerNorm(embeddings)

	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		t = checkpoint.checkpoint(self.ln2, x, use_reentrant=False)
		x = x + self.ffwd(self.ln2(t))
		return x


class FeedForward(nn.Module):
	def __init__(self, embeddings, d_chance=0.2):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(embeddings, 4*embeddings),
			nn.ReLU(),
			nn.Linear(4*embeddings, embeddings),
			nn.Dropout(d_chance))

	def forward(self, x):
		return self.net(x)

class Model(nn.Module):
	def __init__(self, sentence_data, embed_dims, heads=4, blocks=4):
		super().__init__()
		self.context_size = sentence_data[0]
		self.vocab_size = sentence_data[1]
		self.dims = embed_dims


		self.token_embedding_table = nn.Embedding(self.vocab_size, embed_dims)
		self.position_embedding_table = nn.Embedding(self.context_size, embed_dims)

		self.blocks = nn.Sequential(*[
			Block(
				embed_dims, 
				heads, 
				self.context_size) 
			for _ in range(blocks)
			]
		)
		self.layer_norm = nn.LayerNorm(self.dims)
		self.lm_head = nn.Linear(self.dims, self.vocab_size)

	def get_batch(self, data, count=8):
		ran = torch.randint(len(data) - self.context_size, (count,))
		batch_in = torch.stack([data[i:i+self.context_size] for i in ran])
		batch_out = torch.stack([data[i+1:i+self.context_size+1] for i in ran])
		batch_in, batch_out = batch_in.to(device), batch_out.to(device)
		return batch_in, batch_out

	def forward(self, idx, targets=None):
		ba, wi = idx.shape
		token_embed = self.token_embedding_table(idx)
		pos_emb = self.position_embedding_table(torch.arange(wi, device=device))
		tok_pos = token_embed + pos_emb
		tok_pos = self.blocks(tok_pos)
		tok_pos = self.layer_norm(tok_pos)
		logits = self.lm_head(tok_pos)
		if None == targets:
			loss = None
		else:
			ba, wi, ch = logits.shape
			logits = logits.view(ba * wi, ch)
			targets = targets.view(ba*wi)
			loss = F.cross_entropy(logits, targets)
		return logits, loss

	def get_distribution(self, idx):
		indx = idx[:, -self.context_size:]
		logits, _ = self(indx)
		logits = logits[:, -1, :]
		distrib = F.softmax(logits, dim=-1)
		return distrib

	def predict(self, idx, maximal_tokens):
		for i in range(maximal_tokens):
			print(f"\r\tPredicting...\t{100*(i+1)/maximal_tokens:.2f}%.", end='')
			distrib = self.get_distribution(idx)
			idx_next = torch.multinomial(distrib, num_samples=1)
			idx = torch.cat((idx, idx_next), dim=1)
		print(f"\r\tPredicting...\tDone.")
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


def proprocess_input(txt):
	raw = txt.split()
	words = len(set(raw))
	print("Encoding data")
	raw = text.split()
	print(f'{len(raw)} total tokens')
	words = list(set(raw))
	toid = {wrd:i for i, wrd in enumerate(words)}
	tostr = {i:wrd for i, wrd in enumerate(words)}
	data = torch.tensor(encode(raw, toid), dtype=torch.long)
	length = len(data)
	print(f'{length} tokens in total')
	training = data[:int(length * 0.9) ]
	print(f'\t{len(training)} training data points')
	validation = data[int(length * 0.9):]
	print(f'\t{len(validation)} validation data points')
	return (len(words), training, validation, toid, tostr)


if __name__ == '__main__':
	print('Beginning...')
	print('Running on',device)
	with open('input.txt', 'r', encoding='utf-8') as f:
		text = f.read()
	print(f'Read text [{len(text)}]')	
	
	vocab_size, training, validation, toid, tostr = proprocess_input(text)

	batch_count = 64
	context_size = 200  #context
	embedding_dims = 200
	meta_size = (context_size, vocab_size)
	learn = 3e-4
	block_count = 16
	head_count = 16

	iters = 10000
	loss_interval = iters // 20
	x = []
	y = []
	print('Setup variables and lambdas.')


	m = Model(meta_size, embedding_dims, heads=head_count, blocks=block_count)
	m = m.to(device)


	optimizer = torch.optim.AdamW(m.parameters(), lr=learn)


	print(f'Setup model and optimization {sum(p.numel() for p in m.parameters())/1e6} M parameters')
	for steps in range(iters):
		if(steps % loss_interval == 0):
			print(f'\rTraining...\t{100*(steps+0.25)/iters:.4f}% \tEvaluating loss sample', end='')
			x.append(steps//loss_interval)
			y.append(m.estimate_loss(20, training))
		print(f'\rTraining...\t{100*(steps+1/3)/iters:.4f}% \tGetting batch', ' ' * 32, end='')
		batch_in, batch_out = m.get_batch(training, batch_count)
		print(f'\rTraining...\t{100*(steps+2/3)/iters:.4f}% \tForwarding through batch', end='')
		output, loss = m(batch_in, batch_out)
		print(f'\rTraining...\t{100*(steps+1)/iters:.4f}% \tOptimizing',' ' * 32, end='')
		
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()
		

	print(f'\rTraining...\t 100%\tDone',' ' *32)
	con = torch.zeros((1,1), dtype=torch.long, device=device)
	pred = m.predict(con, maximal_tokens=500)[0].tolist()
	print(f'\n\nWith a loss of {m.estimate_loss(300, validation)} we have: \n{" ".join(decode(pred, tostr))}')
	plt.plot(x,y)
	plt.show()
	torch.save(m.state_dict(), 'model.ml')

