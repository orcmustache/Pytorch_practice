#%%
import torch
from torch.nn import Linear
import matplotlib.pyplot as plt
#%%
torch.manual_seed(1)
model = Linear(in_features=1,out_features=1)
print(model.bias,model.weight)
# %%
x=torch.tensor([[2.0],[3.3]])
print(model(x)) ### equal to y = weight * x + bias 
# %%
