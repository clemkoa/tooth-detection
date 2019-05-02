import json
import os

def merge_datasets(dataset, output_folder):
    folder = '/Users/clementjoudet/Desktop/perso/tooth-detection/dataset/'
    json_file_1 = open(os.path.join(folder, 'root', dataset + '.json'))
    data1 = json.load(json_file_1)
    json_file_2 = open(os.path.join(folder, 'apical_lesion', dataset + '.json'))
    data2 = json.load(json_file_2)

    all_frames = data2['frames'].copy()
    for frame, bboxes in data1['frames'].items():
        if frame not in all_frames.keys():
            print(frame, 'not in data2')
        all_frames[frame] = all_frames.get(frame, []) + bboxes

    all_input_tags = ','.join(data1['inputTags'].split(',') + data2['inputTags'].split(','))

    result = data1.copy()
    result['frames'] = all_frames
    result['inputTags'] = all_input_tags

    output_path = os.path.join(output_folder, dataset + '.json')
    with open(output_path, 'w') as outfile:
        json.dump(result, outfile)

output_folder = '/Users/clementjoudet/Desktop/tooth/'
datasets = ['gonesse_x67', 'gonesse_x97', 'gonesse_x102', 'google_x90',
            'iran_x116', 'rotschild_x200', 'brazil_cat8', 'brazil_cat10']

for dataset in datasets:
    print(dataset)
    merge_datasets(dataset, output_folder)
