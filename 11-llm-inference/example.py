"""PyTorch 시리즈 11장 예제 — PyTorch 시리즈 11장 — LLM 추론을 코드로: prefill·decode·KV cache
블로그: https://kjoh94.github.io/blog/2026/10/25/pytorch-11-llm-inference/
"""
import torch, torch.nn.functional as F, time
torch.manual_seed(0)
V, D, H = 50, 32, 4          # 어휘, 임베딩, 헤드 수
hd = D // H

class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, D); self.pos = torch.nn.Embedding(128, D)
        self.qkv = torch.nn.Linear(D, 3 * D); self.o = torch.nn.Linear(D, D)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 4 * D), torch.nn.GELU(), torch.nn.Linear(4 * D, D))
        self.head = torch.nn.Linear(D, V)

    def forward(self, ids, start=0, cache=None):
        T = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(start, start + T))
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = [t.view(1, T, H, hd).transpose(1, 2) for t in (q, k, v)]
        if cache is not None:                       # KV cache: 과거 K,V 를 앞에 붙인다
            if cache["k"] is not None:
                k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        att = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        x = x + self.o(att.transpose(1, 2).reshape(1, T, D))
        x = x + self.mlp(x)
        return self.head(x)

model = TinyDecoder().eval()
prompt = torch.tensor([[1, 7, 3, 9, 4]])


# ----------
@torch.no_grad()
def generate_naive(ids, n):
    for _ in range(n):
        logits = model(ids)                                # 매번 전체
        ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
    return ids


# ----------
@torch.no_grad()
def generate_cached(ids, n):
    cache = {"k": None, "v": None}
    logits = model(ids, 0, cache)                          # prefill
    out = ids
    for _ in range(n):
        nxt = logits[:, -1:].argmax(-1)
        out = torch.cat([out, nxt], 1)
        logits = model(nxt, out.shape[1] - 1, cache)       # decode: 토큰 1개
    return out

a = generate_naive(prompt, 20); b = generate_cached(prompt, 20)
print("같은 결과:", torch.equal(a, b))
print(a.tolist())
for name, fn in [("naive", generate_naive), ("kv cache", generate_cached)]:
    t = time.perf_counter(); fn(prompt, 60)
    print(f"{name:8s} {(time.perf_counter() - t) * 1e3:.1f} ms")


# ----------
T = 4
mask = torch.full((T, T), float("-inf")).triu(1)
print(mask)
print(torch.softmax(torch.zeros(T, T) + mask, -1))   # 각 행: 과거에 균등 분배


# ----------
MAXLEN = 16
k_cache = torch.zeros(1, H, MAXLEN, hd)
def write_step(k_new, pos):
    k_cache[:, :, pos:pos + 1] = k_new                     # 위치에 써넣기 (in-place)
    valid = torch.arange(MAXLEN) <= pos
    return torch.where(valid, 0.0, float("-inf"))          # 빈칸 마스크 (shape 고정)
for p in range(3):
    m = write_step(torch.randn(1, H, 1, hd), p)
print("항상 같은 shape:", k_cache.shape, m.tolist())
