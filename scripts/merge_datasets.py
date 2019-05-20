import json
import os

def merge_datasets(dataset, folder1, folder2, output_folder):
    json_file_1 = open(os.path.join(folder1, dataset + '.json'))
    data1 = json.load(json_file_1)
    json_file_2 = open(os.path.join(folder2, dataset + '.json'))
    data2 = json.load(json_file_2)

    all_frames = data2['frames'].copy()
    for frame, bboxes in data1['frames'].items():
        if 'png' not in frame and 'jpg' not in frame:
            frame = str(int(frame) + 1) + '.png'
        if frame not in all_frames.keys():
            print(frame, 'not in data2')
        all_frames[frame] = all_frames.get(frame, []) + bboxes

    all_input_tags = ','.join(data1['inputTags'].split(',') + data2['inputTags'].split(','))

    result = data1.copy()
    result['frames'] = all_frames
    result['inputTags'] = all_input_tags
    result['visitedFrames'] = data2['visitedFrames']

    output_path = os.path.join(output_folder, dataset + '.json')
    json_file_1.close()
    json_file_2.close()
    with open(output_path, 'w') as outfile:
        json.dump(result, outfile)

def main():
    for dataset in datasets:
        print(dataset)

        folder1 = '/Users/clementjoudet/Desktop/perso/tooth-detection/dataset/root'
        folder2 = '/Users/clementjoudet/Desktop/perso/tooth-detection/dataset/apical_lesion'
        merge_datasets(dataset, folder1, folder2, output_folder)

        folder1 = '/Users/clementjoudet/Desktop/perso/tooth-detection/dataset/implant-restoration-endodontic'
        merge_datasets(dataset, folder1, output_folder, output_folder)

output_folder = '/Users/clementjoudet/Desktop/perso/tooth-detection/dataset/root-implant-restoration-endodontic-apical_lesion'
datasets = ['gonesse_x67', 'gonesse_x97', 'gonesse_x102', 'google_x90',
            'iran_x116', 'rothschild_x200', 'brazil_cat8', 'brazil_cat10']
main()
