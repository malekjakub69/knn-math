import os
import numpy as np
import re

def main():
    '''
    Expected structure:
    knn-math/
    ├── app/
    ├── dataset/
    │   ├── train_labels.txt
    │   ├── train_images/
    │   │   ├── img1.png
    │   │   ├── img2.png
    │   │   ├── img3.png
    │   │   ├── ...
    ├── dataset_sort/
    │   ├── sort.py
    '''

    path = '../dataset/'            # Path to the dataset folder
    file_name = 'train_labels.txt'  # File with img names and labels 
    folder_name = 'train_images'    # Name of images folder

    sort_dataset(path, file_name, folder_name)


def sort_dataset(folder_path, labels_name, images_folder):
    with open(folder_path + labels_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    line_stats = np.zeros(10)
    filtered_lines = []

    # filtered_lines = [line for line in lines if '\\' in line]
    for line in lines:
        count = len(re.findall(r'\\\\', line))
        if count > 8:
            line_stats[-1] += 1
        else:
            if count == 0:
                filtered_lines.append(line)
            line_stats[count] += 1

    new_file_name = 'train_labels_oneline.txt'

    with open(folder_path + new_file_name, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)

    print("Line stats:")
    for idx, i in enumerate(line_stats[:-1]):
        if i != 0:
            print(f"{idx+1} lines:\t{i}")
    if line_stats[-1] != 0:
        print(f"Incorrectly classified:\t{line_stats[-1]}")
    print(f"Total:\t{len(lines)}")
    print(len(lines))


if __name__ == '__main__':
    main()