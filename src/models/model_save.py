import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import torch
from src.models.cnn import CNN

if __name__ == '__main__':
    model_path = './models/best_ApproxReLU_model.pth'
    
    input_length = 30
    activation = 'ApproxReLU'
    model = CNN(input_length, activation).to(torch.device('cpu'))
    model.load_state_dict(torch.load(model_path))
    
    model_dict = {}
    for name, param in model.named_parameters():
        model_dict[name] = param.detach().cpu().numpy().tolist()

    json_path = './go/models/best_ApproxReLU_model.json'
    with open(json_path, 'w') as json_file:
        json.dump(model_dict, json_file)

    print(f"Model parameters and layer names saved to {json_path}")
