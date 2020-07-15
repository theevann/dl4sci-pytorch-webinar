torch.arange(5, 20, 2)

(X**2).sum(dim=1).sqrt()

X.logsumexp(0) or X.exp().sum(0).log()

X[2, 1::2]

5*torch.ones(5,5) + 2*torch.eye(5)

torch.arange(4, 33, 2).view(3, 5)

torch.arange(2, 10, 2).view(4, 1).expand(4, 5)

(X!=0).sum().item()

X[X > int(X.float().mean())]

X[X > 3] = 3

if torch.cuda.is_available():
    x.cuda()
