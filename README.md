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
| 6 | [`06-graph-capture`](06-graph-capture/) | [6. 그래프 캡처 — FX·Dynamo·graph break](https://kjoh94.github.io/blog/2026/10/25/pytorch-06-graph-capture/) |
| 7 | [`07-torch-export`](07-torch-export/) | [7. torch.export와 Core ATen IR](https://kjoh94.github.io/blog/2026/10/25/pytorch-07-torch-export/) |
| 8 | [`08-onnx`](08-onnx/) | [8. PyTorch → ONNX](https://kjoh94.github.io/blog/2026/10/25/pytorch-08-onnx/) |
| 9 | [`09-compiler-opt`](09-compiler-opt/) | [9. 컴파일러 최적화 감각 — folding·fusion·메모리](https://kjoh94.github.io/blog/2026/10/25/pytorch-09-compiler-opt/) |
| 10 | [`10-quantization`](10-quantization/) | [10. 양자화, PyTorch 쪽에서](https://kjoh94.github.io/blog/2026/10/25/pytorch-10-quantization/) |
| 11 | [`11-llm-inference`](11-llm-inference/) | [11. LLM 추론을 코드로 — prefill·decode·KV cache](https://kjoh94.github.io/blog/2026/10/25/pytorch-11-llm-inference/) |
| 12 | [`12-custom-ops`](12-custom-ops/) | [12. 커스텀 op — torch.library와 extension](https://kjoh94.github.io/blog/2026/10/25/pytorch-12-custom-ops/) |
