import torch
from calculate_fvd import calculate_fvd

NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 30
CHANNEL = 3
SIZE = 64
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda")
device = torch.device("cpu")

import json
result = {}
result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
print(json.dumps(result, indent=4))