#%%
import torch

#%%
x=torch.tensor(2.0,requires_grad=True)
y = 9*x**4+2*x**3+3*x**2+6*x+1
y.backward()
print(x)
print(x.grad)
print(x.grad_fn)
# %%


# Create a tensor with requires_grad=True
x = torch.randn(3, requires_grad=True)  

# Perform some operations
y = x * 2
z = y.mean()

# Calculate gradients
z.backward() 

# Access the gradients
print(x.grad)  # This will print the gradients of z with respect to x
# %%
