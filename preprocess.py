import json
import jsonlines

user_annoy = False
uncased = True
add_description = True

if add_description:
    descriptions = []
    with open('Molweni.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                descriptions.append(line)
    description = ' ; '.join(descriptions) + ' [dialogue] '
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('t5-small')
    # t = tokenizer.tokenize(description)
    # print(t)
    # print(len(t))
    # exit()
else:
    description = ''


def preprocess_data(file_path, output_path, idx_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    lines = []
    s_idx = []
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
            relations[u_id] = set()
        for relation in session['relations']:
            x, y, type = relation['x'], relation['y'], relation['type']
            relations[y].add((x, type))
        for y in relations:
            if y > 0:
                if len(relations[y]) > 0:
                    for (x, type) in relations[y]:
                        if uncased:
                            lines.append({'context': description + ' | '.join(context[:y + 1]).lower(), 'relation': f'T{y}, T{x} : {type}'.lower()})
                            s_idx.append(f'{s_id}|||T{y}, T{x} : {type}'.lower())
                        else:
                            lines.append({'context': description + ' | '.join(context[:y + 1]), 'relation': f'T{y}, T{x} : {type}'})
                            s_idx.append(f'{s_id}|||T{y}, T{x} : {type}')
                else:
                    if uncased:
                        lines.append({'context': description + ' | '.join(context[:y + 1]).lower(), 'relation': f'none'.lower()})
                        s_idx.append(f'{s_id}|||none'.lower())
                    else:
                        lines.append({'context': description + ' | '.join(context[:y + 1]), 'relation': f'none'})
                        s_idx.append(f'{s_id}|||none')

    if 'train' not in file_path:
        assert len(lines) == len(s_idx)
        with jsonlines.open(idx_path, 'w') as writer:
            writer.write_all(s_idx)

    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(lines)



if __name__ == '__main__':
    for split in ['train', 'dev', 'test']:
        file_path = f'Molweni/DP/{split}.json'
        output_path = f'saved_data/data_prompt/{split}.json'
        idx_path = f'saved_data/data_prompt/{split}.idx'
        preprocess_data(file_path, output_path, idx_path)