# PyTorch 시리즈 예제 코드

블로그 [PyTorch 시리즈](https://kjoh94.github.io/blog/2026/10/25/pytorch-series/)의 장별 예제입니다. 각 폴더의 `example.py`는 CPU만으로 실행됩니다 (PyTorch 2.x, 확인: 2.14 CPU).

```bash
pip install -r requirements.txt
cd 01-tensor && python example.py
```

| 장 | 폴더 | 블로그 |
|---|---|---|
| 1 | [`01-tensor`](01-tensor/) | [1. Tensor의 정체 — storage·stride·view](https://kjoh94.github.io/blog/2026/10/25/pytorch-01-tensor/) |
| 2 | [`02-module-state-dict`](02-module-state-dict/) | [2. nn.Module과 state_dict](https://kjoh94.github.io/blog/2026/10/25/pytorch-02-module-state-dict/) |
| 3 | [`03-autograd`](03-autograd/) | [3. Autograd와 추론 모드](https://kjoh94.github.io/blog/2026/10/25/pytorch-03-autograd/) |
| 4 | [`04-aten-dispatcher`](04-aten-dispatcher/) | [4. ATen과 Dispatcher](https://kjoh94.github.io/blog/2026/10/25/pytorch-04-aten-dispatcher/) |
| 5 | [`05-privateuse1`](05-privateuse1/) | [5. 커스텀 디바이스 백엔드 (PrivateUse1)](https://kjoh94.github.io/blog/2026/10/25/pytorch-05-privateuse1/) |
