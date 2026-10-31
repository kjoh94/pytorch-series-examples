"""PyTorch 시리즈 12장 예제 — PyTorch 시리즈 12장 — 커스텀 op: torch.library
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-12-custom-ops/
"""
import torch
from torch import Tensor

@torch.library.custom_op("mylib::scaled_add", mutates_args=())
def scaled_add(a: Tensor, b: Tensor, alpha: float) -> Tensor:
    # 실제 커널 (여기선 CPU 파이썬; 실전에선 C++/CUDA/내 장치 코드)
    return a + alpha * b

x, y = torch.randn(3), torch.randn(3)
print(scaled_add(x, y, 2.0))
print(torch.ops.mylib.scaled_add(x, y, 2.0))     # dispatcher 경유 호출
print(torch.ops.mylib.scaled_add.default._schema)


# ----------
@scaled_add.register_fake
def _(a, b, alpha):
    torch._check(a.shape == b.shape)
    return torch.empty_like(a)

def f(a, b):
    return scaled_add(a, b, 0.5).relu()

cf = torch.compile(f, fullgraph=True, backend="aot_eager")  # 기본 Inductor는 g++ 필요; aot_eager로도 fake 커널 경로를 검증
print("compile:", torch.allclose(cf(x, y), f(x, y)))

class M(torch.nn.Module):
    def forward(self, a, b): return scaled_add(a, b, 0.5)
ep = torch.export.export(M(), (x, y))
print([str(n.target) for n in ep.graph.nodes if n.op == "call_function"])  # mylib.scaled_add 한 덩어리


# ----------
def setup_context(ctx, inputs, output):
    a, b, alpha = inputs
    ctx.alpha = alpha

def backward(ctx, g):
    return g, g * ctx.alpha, None

scaled_add.register_autograd(backward, setup_context=setup_context)

a = torch.randn(3, requires_grad=True); b = torch.randn(3, requires_grad=True)
scaled_add(a, b, 3.0).sum().backward()
print("grad a:", a.grad, "grad b:", b.grad)   # 1, 3


# ----------
res = torch.library.opcheck(scaled_add, (torch.randn(4), torch.randn(4), 2.0))
print(res)


# ----------
@scaled_add.register_kernel("cpu")
def _(a, b, alpha):
    print("  (CPU 전용 커널 호출)")
    return torch.add(a, b, alpha=alpha)
print(scaled_add(torch.ones(2), torch.ones(2), 4.0))


# ---- 12장 초보자 보강: 가장 작은 커스텀 op 등록 → export 그래프에 한 덩어리로 ----
if __name__ == "__main__":
    import torch

    @torch.library.custom_op("easy::shrink", mutates_args=())
    def shrink(x: torch.Tensor, t: float) -> torch.Tensor:
        return torch.where(x.abs() > t, x - t * x.sign(), torch.zeros_like(x))

    @shrink.register_fake
    def _(x, t):
        return torch.empty_like(x)

    x = torch.tensor([-2.0, -0.3, 0.1, 1.5])
    print("shrink:", torch.ops.easy.shrink(x, 0.5))

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.ops.easy.shrink(x * 2, 0.5)

    ep = torch.export.export(M(), (x,))
    print("export graph:", [str(n.target) for n in ep.graph.nodes if n.op == "call_function"])
