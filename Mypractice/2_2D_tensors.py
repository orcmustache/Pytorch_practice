#%%
import torch
one_d = torch.arange(0,9).view(3,-1)
# %%
one_d.dim()
# %%
two_d = torch.arange(18).view(2,3,3)
# %%
two_d2 = torch.arange(18).view(3,2,3)
# %%
two_d[1,1,1]
# %%
two_d[1,1,2]
# %%
two_d[1,2,2]
# %%
two_d2[1,0:2,0:2]
# %%
mata = torch.tensor([0,3,5,5,5,2]).view(2,-1)
matb= torch.tensor([3,4,3,-2,-4,-2]).view(3,-1)
# %%
prod =torch.matmul(mata,matb)
# %%
mata @ matb
# %%
