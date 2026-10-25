"""PyTorch 시리즈 8장 예제 — PyTorch 시리즈 8장 — PyTorch에서 ONNX로
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-08-onnx/
"""
import torch, numpy as np
import onnx, onnxruntime as ort
from torch.export import Dim

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 16)
        self.fc2 = torch.nn.Linear(16, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP().eval()
x = torch.randn(3, 4)
onnx_prog = torch.onnx.export(model, (x,), dynamo=True,
                              input_names=["x"], output_names=["y"],
                              dynamic_shapes={"x": {0: Dim("batch")}})
onnx_prog.save("mlp.onnx")


# ----------
m = onnx.load("mlp.onnx")
onnx.checker.check_model(m)
print("opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
for n in m.graph.node:
    print(f"{n.op_type:10s} {list(n.input)} -> {list(n.output)}")
print("initializers:", [i.name for i in m.graph.initializer])


# ----------
def dims(vi):
    return [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]
print("input :", dims(m.graph.input[0]))     # ['batch', 4]
print("output:", dims(m.graph.output[0]))


# ----------
sess = ort.InferenceSession("mlp.onnx", providers=["CPUExecutionProvider"])
for b in [1, 3, 17]:
    xb = torch.randn(b, 4)
    with torch.no_grad():
        ref = model(xb).numpy()
    out = sess.run(None, {"x": xb.numpy()})[0]
    print(b, out.shape, "max err:", float(np.abs(ref - out).max()))
    assert np.allclose(ref, out, atol=1e-5)


# ----------
class Branchy(torch.nn.Module):
    def forward(self, x):
        return torch.cond(x.sum() > 0, lambda t: t * 2, lambda t: t - 1, (x,))

ep = torch.export.export(Branchy(), (torch.randn(3),))
print([str(n.target) for n in ep.graph.nodes if n.op == "call_function"])  # cond 가 그래프에 남는다
