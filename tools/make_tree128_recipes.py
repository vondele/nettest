"""Regenerate local recipes after pinning the trainer and engine commits."""
from pathlib import Path
import re
import subprocess
root=Path(__file__).resolve().parents[2]
repo=root/'nettest'
source=subprocess.check_output(['git','show','upstream/main:threats.yaml'],cwd=repo,text=True)
trainer=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'nnue-pytorch',text=True).strip()
engine=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'Stockfish',text=True).strip()
for name,keys,factors,control in [('tree128','tree128','tree-depth2:tree-depth4:tree-depth6',False),('tree128-unfactorized','tree128','',False),('tree128-control','piececount:queen:pawnhalf:manyoffi','',True)]:
 s=source.replace('ls-keys: "piececount:queen:pawnhalf:manyoffi"',f'ls-keys: "{keys}"').replace('ls-factors: "piececount+queen:pawnhalf:manyoffi"',f'ls-factors: "{factors}"').replace('ls-remap-data: official-stockfish/master-binpacks/fishpack32.binpack','ls-remap-data: ""')
 s=s.replace('  owner: xu-shawn\n  sha: 289991ed8fe7a926533c153dd7bbeef9ac737161',f'  owner: TonyCongqianWang\n  repo: {root}/nnue-pytorch\n  sha: {trainer}')
 s=re.sub(r'(reference_code: &reference_code\n  owner: official-stockfish\n  sha: )[0-9a-f]+',r'\g<1>3e8fde75412e7c47fc140cd82022a395bc681d92',s)
 replacement=f'testing_code: &testing_code\n  owner: TonyCongqianWang\n  repo: {root}/Stockfish\n  sha: {engine}\n  target: build\n  make_options: ["CPPFLAGS=-DNNUE_EMBEDDING_OFF'+(' -DTREE128_CONTROL' if control else '')+'", "-o", "net"]'
 s=re.sub(r'testing_code: &testing_code\n  owner: official-stockfish\n  sha: [0-9a-f]+\n  target: build',replacement,s)
 prefix='# Derived from upstream/main threats.yaml at 6b75d7968ee9eb0a3cf98fd30db90cce37aedc28.\n# All training hyperparameters and binpacks preserved; only routing/factor/remap and code pins differ.\n# Local repo fields make unpublished commits usable without pushing.\n# 95/5 is analysis weighting. Observe layer_stack_traffic.jsonl; tree acceptance is not mixture control.\n'
 (repo/f'{name}.yaml').write_text(prefix+s)
