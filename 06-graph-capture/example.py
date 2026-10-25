"""PyTorch 시리즈 6장 예제 — PyTorch 시리즈 6장 — 그래프 캡처: FX, Dynamo, torch.compile
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-06-graph-capture/
"""
import torch
from torch.fx import symbolic_trace

class M(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1

gm = symbolic_trace(M())
print(gm.graph)        # placeholder → call_function relu → add → output
print(gm.code)         # 그래프에서 다시 생성된 파이썬 코드

# 그래프 고치기: relu 를 sigmoid 로 바꾸는 작은 패스
for node in gm.graph.nodes:
    if node.op == "call_function" and node.target is torch.relu:
        node.target = torch.sigmoid
gm.recompile()
print(gm(torch.zeros(2)))   # sigmoid(0)+1 = 1.5


# ----------
def f(x):
    if x.sum() > 0:
        return x * 2
    return x

try:
    symbolic_trace(f)
except Exception as e:
    print("FX 실패:", type(e).__name__)


# ----------
import torch._dynamo as dynamo

def g(x):
    y = torch.sin(x)
    if y.sum() > 0:      # 텐서 값에 의존하는 분기 → break
        y = y * 2
    print("hi")          # 파이썬 부수효과 → break
    return torch.cos(y)

e = dynamo.explain(g)(torch.randn(4))
print("graphs:", e.graph_count, "breaks:", e.graph_break_count)
for r in e.break_reasons:
    print(" -", str(r.reason)[:70])


# ----------
from torch._dynamo.testing import CompileCounter

dynamo.reset(); c = CompileCounter()
h = torch.compile(lambda x, n: x * n, backend=c)
for n in [1, 2, 3, 4]:
    h(torch.randn(3), n)
print("int 인자 바뀜 → frames:", c.frame_count)    # 2

dynamo.reset(); c = CompileCounter()
h = torch.compile(lambda x: x * 2, backend=c)
for s in [2, 3, 4, 5]:
    h(torch.randn(s))
print("shape 바뀜 → frames:", c.frame_count)       # 2


# ----------
def my_backend(gm, example_inputs):
    print("=== 받은 그래프 ===")
    print(gm.graph)
    return gm.forward          # 여기서 패턴 치환·융합·코드 생성을 하면 된다

dynamo.reset()
model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
cm = torch.compile(model, backend=my_backend)
print(cm(torch.randn(3, 4)).shape)
