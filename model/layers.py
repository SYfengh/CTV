from torch import nn
from torch.nn import init


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.hidden_dim = opt["hidden_dim"]
        self.num_class = opt["num_class"]
        self.linear = nn.Linear(self.hidden_dim, self.num_class)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight)  # initialize linear layer

    def forward(self, inputs):
        logits = self.linear(inputs)
        return logits
