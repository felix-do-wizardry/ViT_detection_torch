# %%
import numpy as np
import json, time
import pandas as pd

# %%
bit_count = 3
count = int(np.round(2 ** bit_count))
limit = 3
flip_values = [int(np.round(2 ** i)) for i in range(bit_count)]
flip_values
assert 0 <= limit <= count
a = list(np.random.choice(range(count), size=limit, replace=False))
a = [0, 1]
limit = len(a)

a

# %%
paths = sorted([
    {
        'head': v,
        'start': v,
        'path': (v,),
        'steps': 1,
    }
    for v in a
], key=lambda v: v[0])
paths

old_paths = []
for i in range(100):
    new_paths = []
    p = paths[0]
    if len(p['path']) == limit + 1:
        print('done', p)
        break
    for d in [1, -1]:
        for v in flip_values:
            b = p[0] + d * v
            
            if b not in p[1] and 0 <= b < count:
                if b in a:
                    # new_paths.append(tuple([b, (*p[1], b), p[2] + 1]))
                    _path_updated = (*p['path'], b)
                else:
                    # new_paths.append(tuple([b, (*p[1],), p[2] + 1]))
                    _path_updated = (*p['path'], )
                new_paths.append({
                    'head': b,
                    'start': p['start'],
                    'path': _path_updated,
                    'steps': p['steps'] + 1,
                })
    
    paths = list(set(paths[1:] + new_paths))
    paths = sorted(paths, key=lambda v: v[2] - len(v[1]))
    old_paths.append(p)
    # _log.append([*paths])

df = pd.DataFrame(paths)
df

# %%