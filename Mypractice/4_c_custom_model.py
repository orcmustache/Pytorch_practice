#%%
import torch
import torch.nn as nn

#%%
class LR(nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.linear = nn.Linear(in_size,out_size)
    def forward(self,x):
        pred = self.linear(x)
        return pred
#%%
torch.manual_seed(1)
model1 = LR(1,1)
print(model1)
print(list(model1.parameters()))
# %%
x= torch.tensor([[1.0],[2.3]])
t=model1.forward(x)
print(t)

# %%
