"""PyTorch 시리즈 3장 예제 — PyTorch 시리즈 3 — Autograd와 추론 모드: eval·no_grad·inference_mode
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-03-autograd/
"""
import torch
x = torch.tensor(3.0)                       # 입력: grad 필요 없음
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y = w * x + b
L = y ** 2
print(y.grad_fn, L.grad_fn)                 # <AddBackward0> <PowBackward0>
print(L.grad_fn.next_functions)             # 거꾸로 이어진 노드들
print(x.is_leaf, w.is_leaf, y.is_leaf)      # True True False


# ----------
L.backward()
print(w.grad, b.grad)        # tensor(42.) tensor(14.)

# 누적 확인
y = w * x + b; (y ** 2).backward()
print(w.grad)                # tensor(84.) — 42가 또 더해짐
w.grad = None                # 또는 optimizer.zero_grad()


# ----------
import torch.nn as nn
net = nn.Sequential(*[nn.Linear(256, 256) for _ in range(8)])
inp = torch.randn(64, 256)

out = net(inp)
n_saved = 0
fn = out.grad_fn
seen, stack = set(), [fn]
while stack:                         # 그래프를 따라가며 저장된 텐서 수 세기
    f = stack.pop()
    if f is None or f in seen: continue
    seen.add(f)
    n_saved += sum(1 for a in dir(f) if a.startswith("_saved_"))
    stack += [g for g, _ in f.next_functions]
print("grad_fn 노드 수:", len(seen), " 저장 슬롯 수:", n_saved)

with torch.no_grad():
    out2 = net(inp)
print(out2.grad_fn)                  # None — 그래프 없음


# ----------
import time
net = nn.Sequential(nn.Linear(512, 512), nn.Dropout(0.5), nn.Linear(512, 512))
x = torch.randn(256, 512)

net.train()
print("train 모드 두 번 결과 같나?", torch.equal(net(x), net(x)))     # False (dropout)
net.eval()
print("eval 모드 두 번 결과 같나?", torch.equal(net(x), net(x)))      # True
print("eval 모드에서도 grad_fn:", net(x).grad_fn is not None)          # True!

def bench(ctx):
    t = time.perf_counter()
    for _ in range(200):
        with ctx():
            net(x)
    return time.perf_counter() - t
import contextlib
print(f"기본       {bench(contextlib.nullcontext):.3f}s")
print(f"no_grad    {bench(torch.no_grad):.3f}s")
print(f"inference  {bench(torch.inference_mode):.3f}s")


# ----------
lin = nn.Linear(3, 3)
with torch.inference_mode():
    feat = lin(torch.randn(2, 3))
print(feat.is_inference())                      # True
w2 = torch.randn(3, requires_grad=True)
try:
    (feat * w2).sum().backward()
except RuntimeError as e:
    print("에러:", str(e)[:70], "...")
(feat.clone() * w2).sum().backward()            # OK
print(w2.grad)


# ----------
# 직접 미분 규칙 정의하기: autograd.Function
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)     # 그림 3의 '저장'을 직접 하는 셈
        return x * x
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * 2 * x
t = torch.tensor([1., 2., 3.], requires_grad=True)
Square.apply(t).sum().backward()
print(t.grad)                       # tensor([2., 4., 6.])
print(torch.autograd.gradcheck(Square.apply, torch.randn(3, dtype=torch.double, requires_grad=True)))


# ---- 추가 (초보자용 개편, 2026-10-31) ----
print("\n== gradient = 기울기 ==")
w = torch.tensor(2.0, requires_grad=True)
loss = w ** 2
loss.backward()
print(w.grad)
with torch.no_grad():
    w -= 0.1 * w.grad
print(w)

print("\n== 검증 / 추론 패턴 ==")
model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.2), nn.Linear(8, 1))
loss_fn = nn.MSELoss()
x_val, y_val = torch.randn(16, 4), torch.randn(16, 1)
model.eval()
with torch.no_grad():
    val_loss = loss_fn(model(x_val), y_val)
model.train()
print("val_loss:", round(val_loss.item(), 4), "grad_fn:", val_loss.grad_fn)
model.eval()
with torch.inference_mode():
    pred = model(x_val)
print(pred.shape, pred.is_inference())
