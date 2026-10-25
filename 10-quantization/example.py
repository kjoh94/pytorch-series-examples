"""PyTorch 시리즈 10장 예제 — PyTorch 시리즈 10장 — 양자화, PyTorch 쪽에서
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-10-quantization/
"""
import torch
torch.manual_seed(0)

def quant(x, scale, zp, qmin=-128, qmax=127):
    return torch.clamp(torch.round(x / scale) + zp, qmin, qmax).to(torch.int8)

def dequant(q, scale, zp):
    return (q.float() - zp) * scale

x = torch.randn(8)
scale, zp = x.abs().max().item() / 127, 0
q = quant(x, scale, zp)
print("x  :", x)
print("q  :", q)
print("err:", (x - dequant(q, scale, zp)).abs().max().item(), "<= scale/2 =", scale / 2)


# ----------
def affine_params(x, qmin=-128, qmax=127):
    lo, hi = min(x.min().item(), 0.0), max(x.max().item(), 0.0)
    scale = (hi - lo) / (qmax - qmin)
    zp = int(qmin - round(lo / scale))
    return scale, zp

act = torch.relu(torch.randn(1000))            # 0 이상으로 치우친 분포
for name, (s, z) in {"symmetric": (act.abs().max().item() / 127, 0),
                     "affine": affine_params(act)}.items():
    e = (act - dequant(quant(act, s, z), s, z)).abs().mean().item()
    print(f"{name:9s} scale={s:.4f} zp={z:4d} mean err={e:.5f}")

out = torch.cat([act, torch.tensor([50.0])])   # 이상치 하나
s, z = affine_params(out)
print("이상치 포함 scale:", round(s, 4), "→ 해상도 저하")


# ----------
W = torch.randn(4, 16) * torch.tensor([[0.01], [0.1], [1.0], [5.0]])   # 채널별 크기 차이

qt = torch.quantize_per_tensor(W, scale=W.abs().max().item() / 127, zero_point=0, dtype=torch.qint8)
sc = W.abs().amax(dim=1) / 127
qc = torch.quantize_per_channel(W, sc, torch.zeros(4, dtype=torch.long), axis=0, dtype=torch.qint8)
for name, qq in [("per-tensor", qt), ("per-channel", qc)]:
    rel = ((W - qq.dequantize()).abs().amax(dim=1) / W.abs().amax(dim=1))
    print(f"{name:11s} 채널별 상대오차:", [f"{v:.3f}" for v in rel.tolist()])


# ----------
class MinMaxObserver:
    def __init__(self): self.lo, self.hi = float("inf"), float("-inf")
    def __call__(self, t):
        self.lo = min(self.lo, t.min().item()); self.hi = max(self.hi, t.max().item()); return t
    def params(self): return affine_params(torch.tensor([self.lo, self.hi]))

net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU()).eval()
obs = MinMaxObserver()
with torch.no_grad():
    for _ in range(50):                      # 보정 데이터
        obs(net(torch.randn(16, 8)))
print("보정된 activation scale, zp:", obs.params())


# ----------
x = torch.randn(6, requires_grad=True)
s = 0.05
y = torch.fake_quantize_per_tensor_affine(x, s, 0, -128, 127)
y.sum().backward()
print("snap:", (y / s).detach())          # 모두 정수
print("grad (STE):", x.grad)              # 범위 안은 1


# ----------
lin = torch.nn.Linear(16, 4).eval()
xin = torch.randn(32, 16)
Wq = torch.quantize_per_channel(lin.weight.detach(), lin.weight.abs().amax(1).detach() / 127,
                                torch.zeros(4, dtype=torch.long), 0, torch.qint8)
with torch.no_grad():
    ref = lin(xin)
    approx = torch.nn.functional.linear(xin, Wq.dequantize(), lin.bias)
print("int8 가중치 출력 상대오차:", ((ref - approx).norm() / ref.norm()).item())
