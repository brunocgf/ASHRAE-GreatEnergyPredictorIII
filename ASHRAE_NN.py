
import torch.nn as nn

class NN_1(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN_1, self).__init__()
    self.linear = nn.Linear(input_size, num_classes)

  def forward(self, x):
    out = self.linear(x)
    return out


class NN_2(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN_2, self).__init__()
    self.hidden = nn.Linear(input_size, 1000)
    self.linear = nn.Linear(1000,num_classes)

  def forward(self, x):
    out = self.hidden(x)
    out = self.linear(out)
    return out

class NN_3(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN_3, self).__init__()
    self.hidden = nn.Linear(input_size, 1000)
    self.relu = nn.ReLU()
    self.linear = nn.Linear(1000,num_classes)

  def forward(self, x):
    out = self.hidden(x)
    out = self.relu(out)
    out = self.linear(out)
    return out
