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


class ConvUnit(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ConvUnit, self).__init__()

		self.conv = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=9,
			stride=2,
			bias=True
		)

	def forward(self, x):
		# x: [batch_size, in_channels=16, 21, 21]

		h = self.conv(x)
		# h: [batch_size, out_channels=8, 7, 7]

		return h


class PrimaryCaps(nn.Module):
	def __init__(self):
		super(PrimaryCaps, self).__init__()

		self.conv1_out = 16 # out_channels of Conv1, a ConvLayer just before PrimaryCaps
		self.capsule_units = 49
		self.capsule_size = 8

		def create_conv_unit(unit_idx):
				unit = ConvUnit(
					in_channels=self.conv1_out,
					out_channels=self.capsule_size
				)
				self.add_module("unit_" + str(unit_idx), unit)
				return unit

		self.conv_units = [create_conv_unit(i) for i in range(self.capsule_units)]

	def forward(self, x):
		# x: [batch_size, 16, 21, 21]
		batch_size = x.size(0)

		u = []
		for i in range(self.capsule_units):
			u_i = self.conv_units[i](x)
			# u_i: [batch_size, capsule_size=8, 7, 7]

			u_i = u_i.view(batch_size, self.capsule_size, -1, 1)
			# u_i: [batch_size, capsule_size=8, 49, 1]

			u.append(u_i)
		# u: [batch_size, capsule_size=8, 49, 1] x capsule_units=49

		u = torch.cat(u, dim=3)
		# u: [batch_size, capsule_size=8, 49, capsule_units=49]

		u = u.view(batch_size, self.capsule_size, -1)
		# u: [batch_size, capsule_size=8, 2041=49*49]

		u = u.transpose(1, 2)
		# u: [batch_size, 2041, capsule_size=8]

		u_squashed = squash(u, dim=2)
		# u_squashed: [batch_size, 2041, capsule_size=8]

		return u_squashed


def squash(s, dim):
	# This is Eq.1 from the paper.
	mag_sq = torch.sum(s**2, dim=dim, keepdim=True)
	mag = torch.sqrt(mag_sq)
	s = (mag_sq / (1.0 + mag_sq)) * (s / mag)

	return s
