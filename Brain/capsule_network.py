#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F


class Conv1(nn.Module):
	def __init__(self):
		super(Conv1, self).__init__()

		self.conv = nn.Conv2d(
			in_channels=1,
			out_channels=16,
			kernel_size=3,
			stride=4,
			bias=True
		)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		# x: [batch_size, 1, 84, 84]

		h = self.relu(self.conv(x))
		# h: [batch_size, 16, 21, 21]

		return h


def squash(s, dim):
	# This is Eq.1 from the paper.
	mag_sq = torch.sum(s**2, dim=dim, keepdim=True)
	mag = torch.sqrt(mag_sq)
	s = (mag_sq / (1.0 + mag_sq)) * (s / mag)

	return s
