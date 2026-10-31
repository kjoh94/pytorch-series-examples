"""PyTorch 시리즈 1장 예제 — PyTorch 시리즈 1 — Tensor의 정체: storage·stride·view·contiguous·dtype·device
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-01-tensor/
"""
import torch

x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
print(x)
print("shape :", x.shape)            # torch.Size([2, 3])
print("stride:", x.stride())         # (3, 1)
print("offset:", x.storage_offset()) # 0
print("dtype :", x.dtype)            # torch.float32
print("device:", x.device)           # cpu
print("storage:", x.untyped_storage().nbytes(), "bytes")  # 6 * 4 = 24


# ----------
x = torch.arange(6.).reshape(2, 3)
i, j = 1, 2
flat = x.storage_offset() + i * x.stride(0) + j * x.stride(1)
print(flat, x[i, j].item())              # 5 5.0
print(x.flatten()[flat].item())          # 5.0  (연속 텐서라서 가능)


# ----------
print(torch.zeros(2, 3, 4).stride())  # (12, 4, 1)


# ----------
x = torch.arange(6.)
y = x.view(2, 3)
z = x.view(3, 2)
print(x.data_ptr() == y.data_ptr() == z.data_ptr())  # True — 같은 메모리
y[0, 0] = 100
print(x[0].item(), z[0, 0].item())                    # 100.0 100.0


# ----------
x = torch.arange(12.).reshape(3, 4)
s = x[1:, 2:]                     # 오른쪽 아래 2×2
print(s.storage_offset(), s.stride())   # 6 (4, 1)  — 1행*4 + 2열 = 6에서 시작
print(s.data_ptr() == x.data_ptr() + 6 * x.element_size())  # True


# ----------
x = torch.arange(6.).reshape(2, 3)
xt = x.t()
print(xt.shape, xt.stride(), xt.is_contiguous())  # (3, 2) (1, 3) False
try:
    xt.view(6)
except RuntimeError as e:
    print("view 실패:", str(e)[:60], "...")


# ----------
x = torch.arange(6.).reshape(2, 3)
xt = x.t()
c = xt.contiguous()
print(c.stride(), c.data_ptr() == x.data_ptr())   # (2, 1) False — 복사됨
print(x.contiguous() is x)                         # True — 이미 연속이면 그대로

r1 = x.reshape(3, 2);  print(r1.data_ptr() == x.data_ptr())   # True  (view로 처리)
r2 = xt.reshape(6);    print(r2.data_ptr() == x.data_ptr())   # False (복사)


# ----------
img = torch.randn(1, 3, 4, 4)
cl = img.contiguous(memory_format=torch.channels_last)
print(img.stride(), cl.stride())   # (48, 16, 4, 1) (48, 1, 12, 3)


# ----------
b = torch.tensor([10., 20., 30.])
e = b.expand(4, 3)
print(e.stride(), e.data_ptr() == b.data_ptr())   # (0, 1) True
A = torch.ones(4, 3)
print(A + b)   # 내부적으로 같은 broadcasting 규칙

# 주의: stride 0 텐서에 in-place 쓰기는 금지
try:
    e.add_(1)
except RuntimeError as e_:
    print("in-place 실패:", str(e_)[:50], "...")


# ----------
for dt in [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.bool]:
    t = torch.zeros(1000, dtype=dt)
    print(f"{str(dt):15s} element_size={t.element_size()}  total={t.nbytes} bytes")

print(torch.tensor(70000., dtype=torch.float16))   # inf  — 범위 초과
print(torch.tensor(70000., dtype=torch.bfloat16))  # 70144 — 범위는 OK, 정밀도 손실
print(torch.tensor(0.1, dtype=torch.float16).item(), torch.tensor(0.1, dtype=torch.bfloat16).item())


# ----------
a = torch.ones(3, dtype=torch.float16)
print((a + torch.ones(3)).dtype)        # float32 — 승격!
print((a + 1.0).dtype)                  # float16 — 파이썬 스칼라는 승격 안 시킴


# ----------
dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, device=dev)
y = x.to("cpu")                     # GPU였다면 복사 + 동기화
print(x.device, y.device)

# meta 디바이스: 메모리 없이 shape/dtype만 가진 텐서 — 거대 모델 구조 확인에 유용
m = torch.empty(10_000, 10_000, device="meta")
print(m.shape, m.device)             # 실제 400MB를 할당하지 않는다


# ----------
import torch
def describe(name, t):
    print(f"{name:10s} shape={tuple(t.shape)} stride={t.stride()} off={t.storage_offset()} "
          f"contig={t.is_contiguous()} storage={t.untyped_storage().data_ptr() % 10000}")

x = torch.arange(24.).reshape(2, 3, 4)
describe("x", x)
describe("x[:,1]", x[:, 1])
describe("permute", x.permute(2, 0, 1))
describe("unsqueeze", x.unsqueeze(0))
describe("expand", torch.ones(1, 4).expand(3, 4))
describe("contig", x.permute(2, 0, 1).contiguous())


# ---- 추가 (초보자용 개편, 2026-10-31) ----
print("\n== 0. 메모리는 한 줄 ==")
x = torch.tensor([[0., 1., 2.], [3., 4., 5.]])
print(x.flatten())
print("x[1,2,3] in (2,3,4):", torch.arange(24.).reshape(2, 3, 4)[1, 2, 3].item())

print("\n== 복사 판단 ==")
x = torch.arange(6.).reshape(2, 3)
print((x + 1).data_ptr() == x.data_ptr(), x.t().data_ptr() == x.data_ptr())

print("\n== NHWC -> NCHW ==")
imgs = torch.rand(8, 32, 32, 3)
chw = imgs.permute(0, 3, 1, 2)
print(chw.shape, chw.is_contiguous())
print(torch.nn.Conv2d(3, 16, 3)(chw).shape)
print(chw.reshape(8, -1).shape)

print("\n== 자주 보는 에러 ==")
print(torch.arange(6.).reshape(2, 3).t().reshape(6))
lin = torch.nn.Linear(3, 1)
try:
    lin(torch.randn(2, 3, dtype=torch.float16))
except RuntimeError as e:
    print("dtype 에러:", str(e)[:60])
print(lin(torch.randn(2, 3, dtype=torch.float16).float()).dtype)
