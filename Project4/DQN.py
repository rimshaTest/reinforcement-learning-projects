from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 4)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
class Dueling_DQN(nn.Module):
    def __init__(self):
        super(Dueling_DQN, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        
        self.value_stream = nn.Linear(128, 1)
        
        self.advantage_stream = nn.Linear(128, 4)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        
        # Split into two streams
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        
        # Using Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        Q = V + A - A.mean(dim=-1, keepdim=True)
        
        return Q