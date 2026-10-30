"""기초편 A — 텐서를 숫자 상자로. 실행: python example.py"""
import torch

# 1. 만들기
s = torch.tensor(7.0)
v = torch.tensor([1, 2, 3, 4])
m = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("스칼라:", s, s.ndim)
print("벡터  :", v, v.ndim)
print("행렬  :\n", m, m.ndim)

print(torch.zeros(2, 3))
print(torch.ones(2, 3))
print(torch.arange(6))
torch.manual_seed(0)
print(torch.rand(2, 3))

# 2. shape
print("m.shape:", m.shape, "행:", m.shape[0], "열:", m.shape[1])
print("reshape:", torch.arange(6).reshape(3, 2))

# 3. 인덱싱
print("m[0]   :", m[0])
print("m[:, 1]:", m[:, 1])
print("m[1, 2]:", m[1, 2].item())
print("m[m > 3]:", m[m > 3])

# 4. 연산
a = torch.tensor([1., 2., 3.])
b = torch.tensor([10., 20., 30.])
print("a + b :", a + b)
print("a * 2 :", a * 2)
print("a * b :", a * b)
print("sum   :", a.sum().item(), "mean:", a.mean().item())
print("행 합 :", m.sum(dim=1))
print("브로드캐스팅:\n", m + torch.tensor([100, 200, 300]))

# 5. dtype / device
print("m.dtype:", m.dtype, "a.dtype:", a.dtype)
print("float로:", m.float().dtype)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", a.to(device).device)
