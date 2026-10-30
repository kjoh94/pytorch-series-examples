"""기초편 C — loss와 학습 루프. 실행: python example.py"""
import torch
import torch.nn as nn

torch.manual_seed(0)

# 1. loss 손으로 계산
pred = torch.tensor([2.5, 0.0, 2.0])
true = torch.tensor([3.0, -0.5, 2.0])
print("MSE 손계산:", ((pred - true) ** 2).mean().item())
print("MSE nn    :", nn.MSELoss()(pred, true).item())

# 2. gradient 맛보기: f(w) = w^2, w = 2
w = torch.tensor(2.0, requires_grad=True)
f = w ** 2
f.backward()
print("df/dw at w=2:", w.grad.item())   # 4

# 3. 손으로 경사하강 몇 걸음
w = torch.tensor(2.0, requires_grad=True)
for step in range(5):
    f = w ** 2
    f.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
    w.grad.zero_()
    print(f"step {step}: w={w.item():.4f}  f={f.item():.4f}")

# 4. 직선 맞추기: y = 2x + 1 + 잡음
X = torch.linspace(0, 5, 50).unsqueeze(1)   # (50, 1)
Y = 2 * X + 1 + 0.3 * torch.randn_like(X)

class LineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

model = LineModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

for step in range(501):
    pred = model(X)                 # ① 예측
    loss = loss_fn(pred, Y)         # ② 채점
    optimizer.zero_grad()           # ③ 지난 gradient 지우기
    loss.backward()                 # ④ gradient 계산
    optimizer.step()                # ⑤ 한 걸음
    if step % 100 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}  "
              f"w={model.linear.weight.item():.3f}  b={model.linear.bias.item():.3f}")

# 5. 학습 끝: 예측해 보기
model.eval()
with torch.no_grad():
    print("x=10 예측:", model(torch.tensor([[10.0]])).item(), "(정답 근처 21)")

# 6. 보너스: 2-클래스 분류
A = torch.randn(100, 2) + torch.tensor([2., 2.])
Bc = torch.randn(100, 2) + torch.tensor([-2., -2.])
Xc = torch.cat([A, Bc]); Yc = torch.cat([torch.ones(100), torch.zeros(100)]).unsqueeze(1)
clf = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
opt = torch.optim.SGD(clf.parameters(), lr=0.1)
bce = nn.BCEWithLogitsLoss()
for step in range(200):
    logit = clf(Xc); loss = bce(logit, Yc)
    opt.zero_grad(); loss.backward(); opt.step()
acc = ((clf(Xc) > 0).float() == Yc).float().mean().item()
print(f"분류 loss={loss.item():.4f}  정확도={acc:.2%}")
