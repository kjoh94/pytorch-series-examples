"""PyTorch 시리즈 5장 예제 — PyTorch 시리즈 5 — 커스텀 디바이스 백엔드: PrivateUse1으로 새 장치 꽂기
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-05-privateuse1/
"""
import torch
torch.utils.rename_privateuse1_backend("mydev")

# ④ 장치 모듈: torch.mydev — 실제 백엔드는 C++ 확장과 연결된 함수들을 넣는다
import types
mydev_mod = types.ModuleType("mydev")
mydev_mod.is_available = lambda: False      # 장난감이라 실제 장치는 없음
mydev_mod.device_count = lambda: 0
mydev_mod.current_device = lambda: 0
mydev_mod._is_in_bad_fork = lambda: False
mydev_mod.manual_seed_all = lambda seed: None
torch._register_device_module("mydev", mydev_mod)
print(torch._C._get_privateuse1_backend_name())     # mydev

torch.utils.generate_methods_for_privateuse1_backend()
print(hasattr(torch.Tensor, "mydev"), hasattr(torch.Tensor, "is_mydev"))  # True True
print(torch.device("mydev:0"))                      # 이제 장치 문자열로 인식된다


# ----------
lib = torch.library.Library("aten", "IMPL")
calls = []
def my_add(a, b, alpha=1):
    calls.append("mydev add")
    return a
lib.impl("add.Tensor", my_add, "PrivateUse1")
print(torch._C._dispatch_has_kernel_for_dispatch_key("aten::add.Tensor", "PrivateUse1"))  # True


# ----------
from torch.utils._pytree import tree_map

SUPPORTED = {"aten.add.Tensor", "aten.gelu.default", "aten.native_layer_norm.default", "aten.mul.Tensor", "aten.mm.default", "aten.t.default",
             "aten.addmm.default", "aten.relu.default", "aten.detach.default"}
stats = {"device": 0, "fallback": 0}

class ToyDev(torch.Tensor):
    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_wrapper_subclass(cls, data.shape, dtype=data.dtype,
                                                   strides=data.stride(), device="meta")
    def __init__(self, data): self.data_ = data
    def __repr__(self): return f"ToyDev({self.data_})"
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        name = str(func)
        stats["device" if name in SUPPORTED else "fallback"] += 1
        unwrap = lambda t: t.data_ if isinstance(t, ToyDev) else t
        wrap = lambda t: ToyDev(t) if isinstance(t, torch.Tensor) else t
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {}))
        return tree_map(wrap, out)

x = ToyDev(torch.randn(4, 8))
lin = torch.nn.Linear(8, 3)
w, b = ToyDev(lin.weight.detach()), ToyDev(lin.bias.detach())
y = torch.relu(torch.nn.functional.linear(x, w, b))
y2 = torch.softmax(y, dim=-1)        # softmax 는 '미지원' → fallback 카운트
print(stats)
print(torch.allclose(y2.data_, torch.softmax(torch.relu(lin(x.data_)), -1)))


# ----------
from collections import Counter
from torch.utils._python_dispatch import TorchDispatchMode
class OpCount(TorchDispatchMode):
    def __init__(self): super().__init__(); self.c = Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.c[str(func)] += 1
        return func(*args, **(kwargs or {}))

model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.LayerNorm(64),
                            torch.nn.Linear(64, 10), torch.nn.Softmax(-1)).eval()
inp = torch.randn(2, 32)
with torch.no_grad(), OpCount() as oc:
    model(inp)
# 참고: nn.TransformerEncoderLayer 는 추론 시 '_transformer_encoder_layer_fwd' 같은
# 융합 op 하나로 내려가기도 한다 — 백엔드 입장에선 그 op도 커버하거나 분해시켜야 한다
for op, n in oc.c.most_common():
    mark = "O" if op in SUPPORTED else "-"
    print(f"{mark} {op:45s} x{n}")
print("커버리지:", sum(op in SUPPORTED for op in oc.c), "/", len(oc.c))


# ----------
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    print("내 컴파일러가 받은 그래프 op 수:", len(gm.graph.nodes))
    return gm.forward                  # 여기서 내 장치용 코드로 바꾸면 된다
f = torch.compile(lambda x: torch.relu(x @ x.T) + 1, backend=my_backend)
print(f(torch.randn(4, 4)).shape)


# ---- 추가 (초보자용 개편, 2026-10-31): 1단계 '이름 붙이기'만 해 보기 ----
# 주의: 위쪽 예제에서 이미 이름을 붙였다면 이 블록은 건너뛴다
print("\n== 1단계: 이름 붙이기 ==")
print("현재 이름:", torch._C._get_privateuse1_backend_name())
if torch._C._get_privateuse1_backend_name() == "privateuseone":
    torch.utils.rename_privateuse1_backend("mydev")
name = torch._C._get_privateuse1_backend_name()
print(torch.device(f"{name}:0"))
