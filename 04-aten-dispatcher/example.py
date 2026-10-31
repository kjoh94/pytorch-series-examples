"""PyTorch 시리즈 4장 예제 — PyTorch 시리즈 4 — ATen과 Dispatcher: torch.mm 한 줄이 커널까지 가는 길
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-04-aten-dispatcher/
"""
import torch
a, b = torch.randn(3, 4), torch.randn(4, 5)
print(torch.ops.aten.mm(a, b).shape)          # torch.mm 과 같은 op를 '이름'으로 호출
print(torch.ops.aten.mm.default._schema)     # aten::mm(Tensor self, Tensor mat2) -> Tensor


# ----------
op = torch.ops.aten.add.Tensor
print(op._schema)       # aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
# op 이름의 점 뒤는 overload 이름: add.Tensor, add.Scalar, add.out ...
print([o for o in torch.ops.aten.add.overloads()][:5])

# 이 op에 어떤 키로 커널이 등록되어 있나
for key in ["CPU", "CUDA", "Meta", "AutogradCPU", "PrivateUse1"]:
    print(key, torch._C._dispatch_has_kernel_for_dispatch_key("aten::add.Tensor", key))


# ----------
x = torch.randn(2)
print(torch._C._dispatch_keys(x))
y = torch.randn(2, requires_grad=True)
print(torch._C._dispatch_keys(y))
m = torch.empty(2, device="meta")
print(torch._C._dispatch_keys(m))   # Meta 키


# ----------
a = torch.randn(3, 4, requires_grad=True)
z = torch.mm(a, torch.randn(4, 5))
print(z.grad_fn)                     # <MmBackward0> — Autograd 정거장의 흔적

# 정거장을 직접 확인: 디스패처 전체 등록 표 일부 출력
dump = torch._C._dispatch_dump("aten::mm")
for line in dump.splitlines():
    if line.split(":")[0] in ("CPU", "Meta", "AutogradCPU", "AutocastCPU", "CompositeExplicitAutograd"):
        print(line[:90])


# ----------
# 모든 텐서 생성의 바닥에는 empty 계열 op가 있다
from torch.utils._python_dispatch import TorchDispatchMode
class LogOps(TorchDispatchMode):
    def __init__(self): super().__init__(); self.ops = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))

with LogOps() as log:
    t = torch.zeros(3, 4)
    u = t.t()                # view
    v = u.contiguous()       # 복사 필요
print(log.ops)               # zeros, t, clone 등 — 실제로 내려간 aten op


# ----------
import torch.nn.functional as F
with LogOps() as log:
    F.linear(torch.randn(2, 3), torch.randn(4, 3), torch.randn(4))
print(log.ops[-2:])          # ['aten.t.default', 'aten.addmm.default']

with LogOps() as log:
    F.gelu(torch.randn(3)); torch.nn.functional.silu(torch.randn(3)); F.layer_norm(torch.randn(2, 4), (4,))
print(log.ops)


# ----------
a = torch.empty(3, 4, device="meta")
b = torch.empty(4, 5, device="meta")
c = a @ b
print(c.shape, c.device)                                   # (3, 5) meta

big = torch.nn.Linear(50_000, 50_000, device="meta")       # 10GB 짜리지만 메모리 0
print(big(torch.empty(8, 50_000, device="meta")).shape)


# ----------
lib = torch.library.Library("demo", "DEF")
lib.define("scale2(Tensor x) -> Tensor")
def scale2_cpu(x): return x * 2
def scale2_meta(x): return torch.empty_like(x)
lib.impl("scale2", scale2_cpu, "CPU")
lib.impl("scale2", scale2_meta, "Meta")

print(torch.ops.demo.scale2(torch.ones(3)))
print(torch.ops.demo.scale2(torch.empty(3, device="meta")).device)
print(torch._C._dispatch_has_kernel_for_dispatch_key("demo::scale2", "CPU"))


# ---- 추가 (초보자용 개편, 2026-10-31) ----
print("\n== profiler로 aten op 이름 보기 ==")
import torch.nn as nn
x = torch.randn(2, 4)
lin = nn.Linear(4, 3)
with torch.profiler.profile() as prof:
    lin(x)
print(prof.key_averages().table(row_limit=6))
