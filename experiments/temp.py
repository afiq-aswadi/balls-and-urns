#%%
import torch
resid = torch.tensor([-0.584,-0.70])
W_U0 = torch.tensor([0.795,0.976])
W_U1 = torch.tensor([0.042, 0.961])

logit = torch.dot(resid, W_U1 - W_U0)
print(logit)
#%%
prob = torch.sigmoid(torch.tensor(0.5))  # For binary case, sigmoid is equivalent to softmax
print(f"P(token=1) = {prob:.3f}")

# %%
