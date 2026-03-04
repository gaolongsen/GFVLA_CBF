import torch
from dualarmvla.models.dualarmvla.model_loader import lift3d_clip_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = lift3d_clip_base().to(device)
point_cloud = torch.randn([4, 1024, 3]).to(device)
output = model(point_cloud)

print(output.shape)
