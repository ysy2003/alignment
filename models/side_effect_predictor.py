# models/side_effect_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SideEffectPredictor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=27):
        super(SideEffectPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, molecule_vector):
        x = F.relu(self.fc1(molecule_vector))
        output = self.fc2(x)
        return output
    


