#%% Into Sensors
import numpy as np
import matplotlib.pyplot as plt
np.__version__

# %%
import torch
v = torch.tensor([1,2,3])
print(v[2])
print(v[1:])
# %%
f = torch.FloatTensor([1,2,3])
f.size()
# %%
f1 = torch.FloatTensor([1,2,3,4,5,6])
print(f1)
print(f1.view(3,2))
print(f1.size())
print(f1.view(3,2).size())
print(f1.view(2,-1))
# %%
a= np.array([1,2,3,4,5])
t1 = torch.from_numpy(a)
# %%
a2= np.array([1,2,3,4,5],dtype=np.float64)
# %%
t_one = torch.tensor([1,2,3])
t_two = torch.tensor([2,4,5])
sum1= t_one +t_two
prod1 = torch.dot(t_one,t_two)
# %%
torch.linspace(0,1,100)
# %%
torch.linspace(0,10,5)
# %%
x = torch.linspace(0,10,100)
y = torch.sin(x)
plt.plot(x,y)
# %%
plt.plot(x.numpy(),y.numpy())
# %%
