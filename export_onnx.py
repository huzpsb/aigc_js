"""
Extract weights from infer.js and export to ONNX.
Network: embedding -> 2x HsCNN blocks -> final_conv -> global_avg_pool -> linear -> sigmoid
"""
import re, json, torch, torch.nn as nn, numpy as np

# ── 1. Parse weights from infer.js ──────────────────────────────────────────

with open("infer.js", "r", encoding="utf-8") as f:
    js = f.read()

def extract_array(name):
    # Match: var NAME = [...]  (flat number array)
    pat = rf'var {re.escape(name)}\s*=\s*\[([\s\S]*?)\];'
    m = re.search(pat, js)
    if not m:
        raise ValueError(f"Cannot find {name}")
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', m.group(1))
    return [float(x) for x in nums]

# All weight variable names
weight_names = [
    "W_emb", "B_emb",
    "B1_dw1_w", "B1_dw1_b", "B1_dw2_w", "B1_dw2_b",
    "B1_dw3_w", "B1_dw3_b", "B1_dw4_w", "B1_dw4_b",
    "B1_pw1_1_w", "B1_pw1_1_b", "B1_pw1_2_w", "B1_pw1_2_b",
    "B1_pw1_3_w", "B1_pw1_3_b", "B1_pw2_w", "B1_pw2_b",
    "B2_dw1_w", "B2_dw1_b", "B2_dw2_w", "B2_dw2_b",
    "B2_dw3_w", "B2_dw3_b", "B2_dw4_w", "B2_dw4_b",
    "B2_pw1_1_w", "B2_pw1_1_b", "B2_pw1_2_w", "B2_pw1_2_b",
    "B2_pw1_3_w", "B2_pw1_3_b", "B2_pw2_w", "B2_pw2_b",
    "W_final_conv", "B_final_conv",
    "W_lin1", "B_lin1", "W_lin2", "B_lin2",
]

print("Extracting weights...")
W = {}
for name in weight_names:
    W[name] = extract_array(name)
    print(f"  {name}: {len(W[name])} values")

# Also extract TOKENS string for the tokenizer
tok_match = re.search(r'var TOKENS\s*=\s*"(.*?)"', js)
TOKENS = tok_match.group(1)
print(f"  TOKENS: {len(TOKENS)} chars")

# ── 2. Build PyTorch model matching the JS architecture ─────────────────────

class HsCNNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Depthwise convolutions (groups=channels)
        self.dw1 = nn.Conv1d(32, 32, 3, padding=1, groups=32)
        self.dw2 = nn.Conv1d(16, 16, 9, padding=4, groups=16)
        self.dw3 = nn.Conv1d(8, 8, 27, padding=13, groups=8)
        self.dw4 = nn.Conv1d(8, 8, 81, padding=40, groups=8)
        # Pointwise convolutions
        self.pw1_1 = nn.Conv1d(64, 16, 1)
        self.pw1_2 = nn.Conv1d(64, 16, 1)
        self.pw1_3 = nn.Conv1d(128, 32, 1)
        self.pw2 = nn.Conv1d(64, 64, 1)

    def forward(self, x):
        out1 = self.dw1(x[:, :32, :])
        out2 = self.dw2(x[:, 32:48, :])
        out3 = self.dw3(x[:, 48:56, :])
        out4 = self.dw4(x[:, 56:64, :])
        step1 = torch.cat([out1, out2, out3, out4], dim=1)
        y1 = self.pw1_1(x)
        y2 = self.pw1_2(step1)
        y3 = self.pw1_3(torch.cat([x, step1], dim=1))
        step2 = torch.cat([y1, y2, y3], dim=1)
        out = self.pw2(step2)
        return nn.functional.leaky_relu(out, 0.01)


class AiDetModel(nn.Module):
    """
    Embedding is a special conv1d over one-hot token indices.
    We implement it as nn.Embedding + reshape, but to match the JS exactly
    we need the custom embed logic. For ONNX export we'll use a gather-based approach.

    Actually, the JS embed is: for each output channel, for each time step,
    sum over kernel_size=3 window: weight[oc * vocab_size * 3 + token[t+k-1] * 3 + k] + bias[oc]

    This is equivalent to: Embedding(vocab, out_ch*3) reshaped, then a grouped sum.
    But simplest: just replicate the logic with a standard Embedding lookup + conv.

    Simpler approach: treat embedding as a lookup table W_emb[oc, token, k] with conv1d k=3.
    We can represent this as: one-hot(tokens) -> conv1d(vocab, out_ch, k=3).
    """
    def __init__(self, vocab_size=1999, embed_dim=64):
        super().__init__()
        # Embedding as conv1d over one-hot: weight shape [out_ch, vocab_size, 3]
        self.embed_conv = nn.Conv1d(vocab_size, embed_dim, 3, padding=1)
        self.block1 = HsCNNBlock()
        self.block2 = HsCNNBlock()
        self.final_conv = nn.Conv1d(64, 64, 3, padding=1)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 1)

    def forward(self, token_ids):
        # token_ids: [batch, seq_len] int64
        # One-hot encode
        x_onehot = torch.zeros(token_ids.shape[0], 1999, token_ids.shape[1],
                               dtype=torch.float32, device=token_ids.device)
        x_onehot.scatter_(1, token_ids.unsqueeze(1), 1.0)
        x = self.embed_conv(x_onehot)
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_conv(x)
        x = nn.functional.leaky_relu(x, 0.01)
        x = x.mean(dim=2)  # global avg pool
        x = self.lin1(x)
        x = nn.functional.leaky_relu(x, 0.01)
        x = self.lin2(x)
        return torch.sigmoid(x)


# ── 3. Load weights into model ──────────────────────────────────────────────

model = AiDetModel()

def t(arr, shape=None):
    tensor = torch.tensor(arr, dtype=torch.float32)
    if shape:
        tensor = tensor.reshape(shape)
    return tensor

sd = model.state_dict()

# Embedding conv: JS weight is flat [out_ch * vocab_size * k_size]
# PyTorch Conv1d weight shape: [out_channels, in_channels, kernel_size]
# JS indexing: oc * vocab_size * 3 + token * 3 + k
# So flat layout is [oc, token, k] which matches [out_ch, in_ch=vocab, k=3]
sd['embed_conv.weight'] = t(W['W_emb'], (64, 1999, 3))
sd['embed_conv.bias'] = t(W['B_emb'], (64,))

def load_block(prefix, block_name):
    # Depthwise convs: groups=ch, so weight shape [ch, 1, k]
    for i, (ch, k) in enumerate([(32,3),(16,9),(8,27),(8,81)], 1):
        sd[f'{block_name}.dw{i}.weight'] = t(W[f'{prefix}_dw{i}_w'], (ch, 1, k))
        sd[f'{block_name}.dw{i}.bias'] = t(W[f'{prefix}_dw{i}_b'], (ch,))
    # Pointwise convs
    for name, (in_ch, out_ch) in [('pw1_1',(64,16)),('pw1_2',(64,16)),('pw1_3',(128,32)),('pw2',(64,64))]:
        sd[f'{block_name}.{name}.weight'] = t(W[f'{prefix}_{name}_w'], (out_ch, in_ch, 1))
        sd[f'{block_name}.{name}.bias'] = t(W[f'{prefix}_{name}_b'], (out_ch,))

load_block('B1', 'block1')
load_block('B2', 'block2')

# Final conv: [64, 64, 3] but groups=1, so JS flat is [oc * in_per_group * k + ic * k + k_idx]
# = [out_ch, in_ch, k] which matches PyTorch
sd['final_conv.weight'] = t(W['W_final_conv'], (64, 64, 3))
sd['final_conv.bias'] = t(W['B_final_conv'], (64,))

# Linear layers: JS W is [out_dim, in_dim] flat, same as PyTorch
sd['lin1.weight'] = t(W['W_lin1'], (32, 64))
sd['lin1.bias'] = t(W['B_lin1'], (32,))
sd['lin2.weight'] = t(W['W_lin2'], (1, 32))
sd['lin2.bias'] = t(W['B_lin2'], (1,))

model.load_state_dict(sd)
model.eval()

# ── 4. Quick sanity check ──────────────────────────────────────────────────

# Tokenize a test string using the same logic as JS
def tokenize(text):
    tokens = []
    i = 0
    while i < len(text):
        found = False
        for length in [4, 3, 2, 1]:
            if i + length <= len(text):
                sub = text[i:i+length]
                idx = TOKENS.find(sub)
                if idx >= 0:
                    # In JS: idx is char position, token id = idx + 1 (1-indexed)
                    # Actually let me check the JS getTokens logic
                    tokens.append(idx + 1)
                    i += length
                    found = True
                    break
        if not found:
            i += 1
    return tokens

test_text = "这是一个测试文本，用来验证模型是否正确导出。"
tokens = tokenize(test_text)
print(f"\nTest tokens: {tokens[:20]}...")
input_tensor = torch.tensor([tokens], dtype=torch.int64)
with torch.no_grad():
    score = model(input_tensor).item()
print(f"Test score: {score:.6f}")

# ── 5. Export ONNX ──────────────────────────────────────────────────────────

print("\nExporting ONNX...")
dummy = torch.tensor([tokens], dtype=torch.int64)
torch.onnx.export(
    model, dummy,
    "model.onnx",
    input_names=["token_ids"],
    output_names=["score"],
    dynamic_axes={"token_ids": {0: "batch", 1: "seq_len"}},
    opset_version=17,
)
print("Exported model.onnx")

# Also save TOKENS as a JSON file for the frontend
with open("tokens.json", "w", encoding="utf-8") as f:
    json.dump(TOKENS, f, ensure_ascii=False)
print("Saved tokens.json")
