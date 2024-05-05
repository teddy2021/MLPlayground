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

	def DPO_loss_batch(self, accepted, rejected):

		active_acc, frozen_acc = accepted
		active_rej, frozen_rej = rejected

		rewards_acc = self.beta * (active_acc - frozen_acc).detatch()
		rewards_rej = self.beta * (active_rej - frozen_rej).detatch()

		accepted_ratio = active_acc - frozen_acc
		rejected_ratio = active_rej - active_rej

		losses = F.logsigmoid(self.beta * accepted_ratio) - F.logsigmoid(self.beta * rejected_ratio)


		return losses, rewards_acc, rewards_rej

	def get_loss(self, batch):

		acc_logits, act_acc, act_rej = self.active(batch)[0], self.active.get_distribution(batch)
		fro_logits, fro_acc, fro_rej = self.frozen(batch), self.frozen.get_distribution(batch)
		accepted = (act_acc, fro_acc)
		rejected = (act_rej, fro_rej)
		loss, rewards_acc, rewards_rej = self.DPO_loss_batch(accepted, rejected)

		mean_loss = loss.mean()
		mean_accepted = rewards_acc.mean()
		mean_rejected = rewards_rej.mean()

		return mean_loss, mean_accepted, mean_rejected

if __name__ == "__main__":
	

