import sys,types,collections
sys.modules['yaml']=types.ModuleType('yaml'); sys.path.insert(0,'~/ai/llama.cpp/gguf-py')
from gguf import GGUFReader
r=GGUFReader(sys.argv[1])
for k in r.fields:
  if any(s in k for s in ['architecture','block_count','nextn','file_type','embedding_length','feed_forward','head_count','full_attention','context_length']):
    f=r.fields[k]; v=f.parts[f.data[0]]
    try: v=bytes(v).decode()
    except: v=list(v)
    print(k,v)
print(collections.Counter(t.tensor_type.name for t in r.tensors))
d=collections.defaultdict(set)
for t in r.tensors:
  n=t.name.split('.'); key='.'.join(n[2:]) if n[0]=='blk' else t.name
  d[t.tensor_type.name].add(key)
for k,v in d.items(): print(k, sorted(v))
