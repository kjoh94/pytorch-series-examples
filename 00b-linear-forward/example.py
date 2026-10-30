"""기초편 B — Linear와 순전파. 실행: python example.py"""
import torch
import torch.nn as nn

# 1. 뉴런 하나를 손으로
x = torch.tensor([2., 1., 3.])
w = torch.tensor([0.5, -1.0, 0.2])
b = torch.tensor(0.1)
y = (w * x).sum() + b
print("뉴런 하나:", y.item())          # 0.5*2 - 1*1 + 0.2*3 + 0.1 = 0.7

# 2. 출력 2개 = 행렬 곱
W = torch.tensor([[1., 0., 1.],
                  [0., 2., 0.]])
print("W @ x:", W @ x)                 # [5, 2]

# 3. 배치로: x @ W.T + b
X = torch.tensor([[2., 1., 3.],
                  [1., 1., 1.]])        # (2개 샘플, 3 특징)
B = torch.tensor([0.0, 1.0])
print("X @ W.T + B:\n", X @ W.T + B)

# 4. nn.Linear와 비교
layer = nn.Linear(3, 2)
print("weight", layer.weight.shape, "bias", layer.bias.shape)
with torch.no_grad():
    layer.weight.copy_(W)
    layer.bias.copy_(B)
print("nn.Linear :\n", layer(X))
print("같은가?", torch.allclose(layer(X), X @ W.T + B))

# 5. 활성화
t = torch.linspace(-3, 3, 7)
print("t      :", t)
print("relu   :", torch.relu(t))
print("sigmoid:", torch.sigmoid(t).round(decimals=2))

# 6. 작은 MLP 순전파 + shape 따라가기
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
inp = torch.randn(4, 3)
h = inp
for i, mod in enumerate(model):
    h = mod(h)
    print(f"{i}: {mod.__class__.__name__:7s} -> {tuple(h.shape)}")
print("파라미터 수:", sum(p.numel() for p in model.parameters()))  # 3*8+8 + 8*1+1 = 41
