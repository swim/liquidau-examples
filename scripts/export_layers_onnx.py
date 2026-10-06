"""
An ONNX export of all-mpnet-base-v2 whose output is every layer's mean-pooled (attention mask),
unit-normalised hidden state: `pooled` [batch, 12, 768] (layer k at index k - 1). Pooling sits inside
the graph, so every runtime computes the same multi-layer features (src/embed-layers.ts reads it).
Also writes an 8-bit copy (dynamic quantisation, uint8 weights) for smaller deployments.

With a texts file (JSON lines with a "text" field) it also checks the export against PyTorch:
  fp32 ONNX vs PyTorch (CPU)   cosine per layer (expected >= 0.9999)
  8-bit ONNX vs fp32 ONNX      cosine per layer (what 8-bit costs the features)

Needs Python with torch, transformers, onnx and onnxruntime.
Usage: export_layers_onnx.py <out-dir> [texts.jsonl] [--max-chars N] [--n 2000]
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

out_dir = sys.argv[1]
texts_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
max_chars = int(sys.argv[sys.argv.index('--max-chars') + 1]) if '--max-chars' in sys.argv else None
n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 2000
os.makedirs(out_dir, exist_ok=True)
name = 'sentence-transformers/all-mpnet-base-v2'
tok = AutoTokenizer.from_pretrained(name)
base = AutoModel.from_pretrained(name).eval()

class Pooled(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask):
        hs = self.m(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        mask = attention_mask.unsqueeze(-1).to(hs[0].dtype)
        pooled = [torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=-1) for h in hs[1:]]
        return torch.stack(pooled, dim=1)

wrapped = Pooled(base).eval()
fp32_path, q8_path = os.path.join(out_dir, 'mpnet-layers.onnx'), os.path.join(out_dir, 'mpnet-layers-q8.onnx')
if not os.path.exists(fp32_path):
    enc = tok(['an example sentence', 'another, longer example sentence here'], padding=True, return_tensors='pt')
    torch.onnx.export(wrapped, (enc['input_ids'], enc['attention_mask']), fp32_path, input_names=['input_ids', 'attention_mask'], output_names=['pooled'],
                      dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'attention_mask': {0: 'batch', 1: 'seq'}, 'pooled': {0: 'batch'}}, opset_version=17, dynamo=False)
    print(f'exported {fp32_path} ({os.path.getsize(fp32_path) / 1e6:.0f} MB)')
if not os.path.exists(q8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(fp32_path, q8_path, weight_type=QuantType.QUInt8)
    print(f'quantised {q8_path} ({os.path.getsize(q8_path) / 1e6:.0f} MB)')

if texts_path is None:
    sys.exit(0)
import onnxruntime as ort
rows = [json.loads(l) for l in open(texts_path)][:n]
texts = [r['text'][:max_chars] if max_chars else r['text'] for r in rows]
sess = {k: ort.InferenceSession(p, providers=['CPUExecutionProvider']) for k, p in (('fp32', fp32_path), ('q8', q8_path))}
out = {'torch': [], 'fp32': [], 'q8': []}
with torch.no_grad():
    for s in range(0, len(texts), 16):
        enc = tok(texts[s:s + 16], padding=True, truncation=True, max_length=384, return_tensors='pt')
        out['torch'].append(wrapped(enc['input_ids'], enc['attention_mask']).numpy())
        feed = {'input_ids': enc['input_ids'].numpy(), 'attention_mask': enc['attention_mask'].numpy()}
        for k in ('fp32', 'q8'): out[k].append(sess[k].run(['pooled'], feed)[0])
out = {k: np.concatenate(v) for k, v in out.items()}
cos = lambda a, b: (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
for label, a, b in (('fp32 ONNX vs PyTorch', 'fp32', 'torch'), ('8-bit vs fp32 ONNX', 'q8', 'fp32')):
    c = cos(out[a], out[b])  # n x 12
    print(f'{label}: ' + ', '.join(f'L{k + 1} min {c[:, k].min():.5f} mean {c[:, k].mean():.5f}' for k in (3, 7, 8, 11)))
np.save(os.path.join(out_dir, 'parity-sample.npy'), out['fp32'][:200])
json.dump(texts[:200], open(os.path.join(out_dir, 'parity-texts.json'), 'w'))
