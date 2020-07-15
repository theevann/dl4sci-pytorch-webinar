# Clearing the gradient
# add this line:
x.grad = None

# Drawing tangent lines
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

f(x, y).backward()
df_dx = x.grad
df_dy = y.grad
