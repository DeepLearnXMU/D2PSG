import json
import jsonlines
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('t5-small')
user_annoy = False
uncased = True
add_description = False

max_edu_num = 20

if add_description:
    descriptions = []
    with open('STAC.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                descriptions.append(line)
    description = ' ; '.join(descriptions) + ' [dialogue] '
else:
    description = ''


def preprocess_data(file_path, output_path, idx_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    lines = []
    s_idx = []
    max_src_len, max_tgt_len = 0, 0
    for s_id, session in enumerate(data):
        context = []
        speaker_dict = {}
        relations = {}
        for u_id, utterance in enumerate(session['edus']):
            speaker = utterance['speaker'].strip()
            if len(speaker.strip()) < 1:
                speaker = 'none'
            if speaker not in speaker_dict:
                if user_annoy:
                    speaker_dict[speaker] = f'S{len(speaker_dict)}'
                else:
                    speaker_dict[speaker] = speaker
            text = utterance['text'].strip()
            edu = f'T{u_id}, {speaker_dict[speaker]}: {text}'
            context.append(edu)
            relations[u_id] = None
        for relation in session['relations']:
            x, y, type = relation['x'], relation['y'], relation['type']
            if relations[y]:
                relations[y].append((x, type))
            else:
                relations[y] = [(x, type)]
        for k in range(1, len(context)):
            beg_idx = max(k + 1 - max_edu_num, 0)
            context_seq = description + ' | '.join(context[beg_idx:k + 1]).lower()
            relation_seq = []
            for k_i in range(beg_idx + 1, k + 1):
                if relations[k_i]:
                    for relation in sorted(relations[k_i], key=lambda x:x[0]):
                        y, x, type = k_i, relation[0], relation[1]
                        if beg_idx <= x < k + 1:
                            relation_seq.append(f'T{y} T{x} {type}'.lower())
            lines.append({'context': context_seq, 'relation': '; '.join(relation_seq) if relation_seq else 'none'})
            s_idx.append(f'{s_id}|||{k}')

        max_src_len = max(max_src_len, len(tokenizer.tokenize(context_seq)))
        max_tgt_len = max(max_tgt_len, len(tokenizer.tokenize('; '.join(relation_seq) if relation_seq else 'none')))
    if 'train' not in file_path:
        assert len(lines) == len(s_idx)
        with jsonlines.open(idx_path, 'w') as writer:
            writer.write_all(s_idx)

    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(lines)
    print(max_src_len, max_tgt_len)


if __name__ == '__main__':
    for split in ['train', 'test']:
        file_path = f'STAC/{split}.json'
        output_path = f'saved_data/data_seq_stac/{split}.json'
        idx_path = f'saved_data/data_seq_stac/{split}.idx'
        preprocess_data(file_path, output_path, idx_path)