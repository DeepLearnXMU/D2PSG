import json
import jsonlines

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
    unavail = 0
    out_of_range = 0
    multi = 0
    avail = 0
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
            if y - x < max_edu_num and y > x:
                relations[y].add((x, type))
                avail += 1
            else:
                if y - x >= max_edu_num:
                    out_of_range += 1
                unavail += 1
        for y in relations:
            if y > 0:
                if len(relations[y]) > 0:
                    if len(relations[y]) > 1:
                        multi += 1
                    for (x, type) in relations[y]:
                        if uncased:
                            lines.append({'context': description + ' | '.join(context[max(0, y - max_edu_num) : y + 1]).lower(), 'relation': f'T{y}, T{x} : {type}'.lower()})
                            s_idx.append(f'{s_id}|||T{y}, T{x} : {type}'.lower())
                        else:
                            lines.append({'context': description + ' | '.join(context[max(0, y - max_edu_num) : y + 1]), 'relation': f'T{y}, T{x} : {type}'})
                            s_idx.append(f'{s_id}|||T{y}, T{x} : {type}')
                else:
                    if uncased:
                        lines.append({'context': description + ' | '.join(context[max(0, y - max_edu_num) : y + 1]).lower(), 'relation': f'none'.lower()})
                        s_idx.append(f'{s_id}|||none'.lower())
                    else:
                        lines.append({'context': description + ' | '.join(context[max(0, y - max_edu_num) : y + 1]), 'relation': f'none'})
                        s_idx.append(f'{s_id}|||none')

    # if 'train' not in file_path:
    #     assert len(lines) == len(s_idx)
    #     with jsonlines.open(idx_path, 'w') as writer:
    #         writer.write_all(s_idx)
    #
    # with jsonlines.open(output_path, 'w') as writer:
    #     writer.write_all(lines)

    print(out_of_range, unavail, multi, avail)


if __name__ == '__main__':
    for split in ['train', 'test']:
        file_path = f'STAC/{split}.json'
        output_path = f'saved_data/data_stac/{split}.json'
        idx_path = f'saved_data/data_stac/{split}.idx'
        preprocess_data(file_path, output_path, idx_path)