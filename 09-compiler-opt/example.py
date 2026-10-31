"""PyTorch 시리즈 9장 예제 — PyTorch 시리즈 9장 — 컴파일러 최적화 감각: folding·fusion·메모리
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-09-compiler-opt/
"""
import torch, time
torch.manual_seed(0)

lin = torch.nn.Linear(8, 4); bn = torch.nn.BatchNorm1d(4)
bn.running_mean.uniform_(-1, 1); bn.running_var.uniform_(0.5, 2)
seq = torch.nn.Sequential(lin, bn).eval()

# BN(y) = (y - mean)/sqrt(var+eps)*gamma + beta  →  y = W x + b 에 접어 넣기
s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
fused = torch.nn.Linear(8, 4)
with torch.no_grad():
    fused.weight.copy_(lin.weight * s[:, None])
    fused.bias.copy_((lin.bias - bn.running_mean) * s + bn.bias)

x = torch.randn(5, 8)
with torch.no_grad():
    print("BN folding 오차:", (seq(x) - fused(x)).abs().max().item())


# ----------
def chain(x):
    return (x.relu() + 1) * 2

x = torch.randn(4_000_000)
fc = torch.compile(chain)          # Inductor 가 한 커널로 합친다
try:
    fc(x)                          # 첫 호출 = 컴파일 (기본 Inductor는 C++ 컴파일러 g++ 필요)
    for name, fn in [("eager", chain), ("compiled", fc)]:
        t = time.perf_counter()
        for _ in range(20): fn(x)
        print(f"{name:9s} {(time.perf_counter() - t) / 20 * 1e3:.2f} ms")
    print("same:", torch.allclose(chain(x), fc(x)))
except Exception as e:
    print("skip torch.compile timing (no C++ compiler?):", type(e).__name__)


# ----------
from torch.fx import symbolic_trace

ELEMENTWISE = {torch.relu, torch.sigmoid, torch.tanh, torch.add, torch.mul,
               "relu", "add", "mul", "sigmoid"}
import operator
ELEMENTWISE |= {operator.add, operator.mul}

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.fc = torch.nn.Linear(4, 4)
    def forward(self, x):
        y = self.fc(x)
        return torch.sigmoid(torch.relu(y) + 1) * 2

gm = symbolic_trace(Net())
groups, cur = [], []
for n in gm.graph.nodes:
    if n.op in ("call_function", "call_method") and n.target in ELEMENTWISE:
        cur.append(n.name)
    else:
        if cur: groups.append(cur); cur = []
if cur: groups.append(cur)
print("fusion 후보 그룹:", groups)    # [['relu', 'add', 'sigmoid', 'mul']]


# ----------
class Deep(torch.nn.Module):
    def forward(self, x):
        a = x.relu(); b = a + 1; c = b * 2; d = c.sigmoid(); return d - 1

gm = symbolic_trace(Deep())
nodes = list(gm.graph.nodes)
order = {n: i for i, n in enumerate(nodes)}
life = {n.name: (order[n], max([order[u] for u in n.users] or [order[n]]))
        for n in nodes if n.op in ("call_function", "call_method")}
free_at, assign = [], {}            # 버퍼별 '비는 시점'
for name, (s, e) in sorted(life.items(), key=lambda kv: kv[1][0]):
    for i, f in enumerate(free_at):
        if f < s:                   # 이미 비었으면 재사용
            assign[name] = i; free_at[i] = e; break
    else:
        assign[name] = len(free_at); free_at.append(e)
print("수명:", life)
print("버퍼 배정:", assign, "→ 버퍼", len(free_at), "개 (텐서", len(life), "개)")


# ---- 9장 초보자 보강: 계산량보다 메모리 훑기 횟수가 시간을 정한다 (g++ 불필요) ----
if __name__ == "__main__":
    import time
    import torch
    torch.set_num_threads(1)
    x = torch.randn(16_000_000)  # 64MB
    def t(f, n=5):
        f(); s = time.perf_counter()
        for _ in range(n): f()
        return (time.perf_counter() - s) / n * 1000
    print(f"add (1 pass)            : {t(lambda: x.add(1)):.1f} ms")
    print(f"sin (1 pass, heavy math): {t(lambda: torch.sin(x)):.1f} ms")
    print(f"x*2+1 -> relu (3 passes): {t(lambda: torch.relu(x * 2 + 1)):.1f} ms")
