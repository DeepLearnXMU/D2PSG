import json

pred_path = 'saved_data/t5-v1_1-large-seq-stac/generated_predictions.txt'
idx_path = 'saved_data/data_seq_stac/test.idx'
stac_path = 'STAC/test.json'

# pred_path = 'saved_data/t5-v1_1-large-seq/generated_predictions.txt'
# idx_path = 'saved_data/data_seq/test.idx'
# stac_path = 'Molweni/DP/test.json'

with open(pred_path, 'r') as f:
    preds = [x.strip() for x in f.readlines()]
with open(idx_path, 'r') as f:
    u_idx = [json.loads(x).strip() for x in f.readlines()]

with open(stac_path, 'r') as f:
    data = json.load(f)

refs = []
hyps = {}
for u_id, dial in enumerate(data):
    _refs = {'t0 t0 none'}
    for rel in dial['relations']:
        x, y, type = rel['x'], rel['y'], rel['type']
        if y == 0 and 't0 t0 none' in _refs:
            _refs -= {'t0 t0 none'}
        _refs.add(f'T{y} T{x} {type}'.lower())
    refs.append(_refs)
    hyps[u_id] = {'t0 t0 none'}

for pred, idx_ref in zip(preds, u_idx):
    s_id, u_id = idx_ref.split('|||')
    s_id = int(s_id)
    if pred == 'none':
        continue
    for _pred in pred.split(';'):
        _pred = _pred.strip()
        if _pred.split()[0] == f't{u_id}':
            hyps[s_id].add(_pred)


# assert len(refs) == len(hyps), (len(refs), len(hyps))

pred_num, ref_num, bi_score, cl_score = 0, 0, 0, 0
for i, ref in enumerate(refs):
    if i not in hyps:
        print(i, ref)
        exit()
    hyp = hyps[i]
    pred_num += len(hyp)
    ref_num += len(ref)
    _ref = {' '.join(y.split()[:2]).strip() for y in ref}
    for x in hyp:
        if x in ref:
            cl_score += 1
        if ' '.join(x.split()[:2]).strip() in _ref:
            bi_score += 1

bi_prec, bi_recall = bi_score/pred_num, bi_score/ref_num
bi_f1 = 2 * bi_prec * bi_recall / (bi_prec + bi_recall)
cl_prec, cl_recall = cl_score/pred_num, cl_score/ref_num
cl_f1 = 2 * cl_prec * cl_recall / (cl_prec + cl_recall)
print(f'Link Accuracy : {bi_f1} ; Link&Rel Accuracy : {cl_f1}')
