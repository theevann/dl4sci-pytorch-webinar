y = f(x)
y.backward()
with torch.no_grad():
    x -= lr * x.grad
x.grad.zero_()

y = f(x)
y.backward()
optimizer.step()
optimizer.zero_grad()
