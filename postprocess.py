import json

pred_path = 'saved_data/t5-v1_1-large-stac/generated_predictions.txt'
idx_path = 'saved_data/data_stac/test.idx'
stac_path = 'STAC/test.json'

# pred_path = 'saved_data/t5-v1_1-large-prompt/generated_predictions.txt'
# idx_path = 'saved_data/data_prompt/test.idx'
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
    _refs = {'t0, t0 : none'} # same as deepsequential, add link to root
    for rel in dial['relations']:
        x, y, type = rel['x'], rel['y'], rel['type']
        if y == 0 and 't0, t0 : none' in _refs:
            _refs -= {'t0, t0 : none'}
        _refs.add(f'T{y}, T{x} : {type}'.lower())
    refs.append(_refs)
    hyps[u_id] = {'t0, t0 : none'}

for pred, idx_ref in zip(preds, u_idx):
    u_id, ref = idx_ref.split('|||')
    u_id = int(u_id)
    if pred == 'none':
        continue
    hyps[u_id].add(pred)


# assert len(refs) == len(hyps), (len(refs), len(hyps))

pred_num, ref_num, bi_score, cl_score = 0, 0, 0, 0
for i, ref in enumerate(refs):
    if i not in hyps:
        print(i, ref)
        exit()
    hyp = hyps[i]
    pred_num += len(hyp)
    ref_num += len(ref)
    _ref = {y.split(':')[0].strip() for y in ref}
    for x in hyp:
        if x in ref:
            cl_score += 1
        if x.split(':')[0].strip() in _ref:
            bi_score += 1

bi_prec, bi_recall = bi_score/pred_num, bi_score/ref_num
bi_f1 = 2 * bi_prec * bi_recall / (bi_prec + bi_recall)
cl_prec, cl_recall = cl_score/pred_num, cl_score/ref_num
cl_f1 = 2 * cl_prec * cl_recall / (cl_prec + cl_recall)
print(f'Link Accuracy : {bi_f1} ; Link&Rel Accuracy : {cl_f1}')
