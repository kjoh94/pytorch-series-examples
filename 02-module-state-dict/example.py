"""PyTorch 시리즈 2장 예제 — PyTorch 시리즈 2 — nn.Module과 state_dict: 모델은 이름표 붙은 텐서 트리다
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-02-module-state-dict/
"""
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.block = nn.Sequential(nn.Linear(8, 2), nn.BatchNorm1d(2))
    def forward(self, x):
        return self.block(self.act(self.fc1(x)))

model = Net()
print(model)
for name, m in model.named_modules():
    print(f"{name or '(root)':10s} {type(m).__name__}")
for name, p in model.named_parameters():
    print(f"{name:16s} {tuple(p.shape)}")
print("총 파라미터 수:", sum(p.numel() for p in model.parameters()))


# ----------
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 2) for _ in range(3)]      # 등록 안 됨!
class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(3))
print(len(list(Bad().parameters())), len(list(Good().parameters())))   # 0 6


# ----------
class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3))                       # 학습 O
        self.register_buffer("scale", torch.tensor(2.0))           # 저장 O, 학습 X
        self.register_buffer("cache", torch.zeros(3), persistent=False)  # 저장 X
        self.plain = torch.zeros(3)                                # 아무것도 아님
    def forward(self, x):
        return x * self.w * self.scale

m = Scale()
print(list(m.state_dict().keys()))           # ['w', 'scale']
print([n for n, _ in m.named_buffers()])     # ['scale', 'cache']
m = m.to(torch.float16)
print(m.w.dtype, m.scale.dtype, m.cache.dtype, m.plain.dtype)  # plain만 float32로 남는다


# ----------
sd = model.state_dict()
for k, v in sd.items():
    print(f"{k:30s} {str(tuple(v.shape)):10s} {v.dtype}")
# 값은 모델 파라미터와 같은 메모리를 공유하는 detach된 텐서다
print(sd["fc1.weight"].data_ptr() == model.fc1.weight.data_ptr())   # True


# ----------
import os, tempfile
path = os.path.join(tempfile.mkdtemp(), "net.pt")
torch.save(model.state_dict(), path)

new = Net()
missing_unexpected = new.load_state_dict(torch.load(path, weights_only=True))
print(missing_unexpected)                   # <All keys matched successfully>
print(torch.equal(new.fc1.weight, model.fc1.weight))  # True

# 키가 다를 때: 'module.' 접두사 제거
wrapped = {"module." + k: v for k, v in model.state_dict().items()}
fixed = {k.removeprefix("module."): v for k, v in wrapped.items()}
print(new.load_state_dict(fixed))

# strict=False 는 '일부만' 로드할 때 (예: 분류 헤드만 다른 경우)
partial = {k: v for k, v in model.state_dict().items() if k.startswith("fc1")}
print(new.load_state_dict(partial, strict=False).missing_keys[:2])


# ----------
class MyLinear(nn.Module):
    """예: 가중치를 fp16으로 들고 있다가 계산 때만 올리는 Linear"""
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.register_buffer("w16", lin.weight.detach().half())
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
    def forward(self, x):
        return nn.functional.linear(x, self.w16.float(), self.bias)

def replace_linear(root: nn.Module):
    for name, child in root.named_children():
        if isinstance(child, nn.Linear):
            setattr(root, name, MyLinear(child))
        else:
            replace_linear(child)          # 재귀로 내려가기

x = torch.randn(5, 4)
model.eval()
ref = model(x)
replace_linear(model)
print(model)
print("최대 오차:", (model(x) - ref).abs().max().item())   # fp16 반올림 정도의 작은 값
print(list(model.state_dict().keys())[:3])   # 키 이름도 바뀌었다: fc1.w16 ...


# ----------
acts = {}
def save_act(name):
    def hook(mod, inp, out):
        acts[name] = (out.shape, out.abs().max().item())
    return hook
model2 = Net().eval()
hs = [m.register_forward_hook(save_act(n)) for n, m in model2.named_modules() if isinstance(m, nn.Linear)]
model2(torch.randn(5, 4))
print(acts)
for h in hs: h.remove()

print(model2.training, model2.block[1].training)   # False False — eval()은 재귀로 전파


# ---- 추가 (초보자용 개편, 2026-10-31) ----
print("\n== Linear 안의 숫자 ==")
lin = nn.Linear(4, 8)
print(lin.weight.shape, lin.bias.shape, sum(p.numel() for p in lin.parameters()))
print(type(lin.weight).__name__, lin.weight.requires_grad)
print(sum(p.numel() for p in nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)).parameters()))  # 17

print("\n== 체크포인트 (학습 이어하기) ==")
import os, tempfile
cpath = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = model(torch.randn(8, 4)).pow(2).mean(); loss.backward(); opt.step()
torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": 3}, cpath)
model2 = Net(); opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
c = torch.load(cpath, weights_only=True)
model2.load_state_dict(c["model"]); opt2.load_state_dict(c["opt"])
print("이어서 시작할 epoch:", c["epoch"] + 1)
