"""PyTorch 시리즈 7장 예제 — PyTorch 시리즈 7장 — torch.export와 ExportedProgram
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-07-torch-export/
"""
import torch
from torch.export import export, Dim

lin = torch.nn.Linear(3, 2)
ep = export(lin, (torch.randn(4, 3),))
print(ep.graph_module.code)    # forward(self, p_weight, p_bias, input): aten.linear ...
print(ep.graph_signature)      # 어떤 입력이 PARAMETER 이고 어떤 게 USER_INPUT 인지
print(list(ep.state_dict.keys()))
print(ep.module()(torch.randn(4, 3)).shape)   # 다시 실행 가능한 모듈로


# ----------
ep2 = ep.run_decompositions()
print([str(n.target) for n in ep2.graph.nodes if n.op == "call_function"])
# linear → permute + addmm 으로 분해됨


# ----------
model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
batch = Dim("batch", min=1, max=64)
ep_dyn = export(model, (torch.randn(4, 3),), dynamic_shapes={"input": {0: batch}})
print(ep_dyn.range_constraints)
m = ep_dyn.module()
for b in [1, 7, 33]:
    print(b, m(torch.randn(b, 3)).shape)

ep_static = export(model, (torch.randn(4, 3),))
try:
    ep_static.module()(torch.randn(8, 3))
except Exception as e:
    print("정적 export 에 배치 8 → 에러:", type(e).__name__)


# ----------
torch.export.save(ep_dyn, "mlp.pt2")
loaded = torch.export.load("mlp.pt2")
print(loaded.module()(torch.randn(5, 3)).shape)
