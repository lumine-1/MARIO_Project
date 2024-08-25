import os

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image


augmentation = A.Compose([
    A.ColorJitter(brightness=0.3, contrast=0.3, p=0.4),
    A.GaussianBlur(blur_limit=(5, 9), sigma_limit=(0.1, 5), p=0.2),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=20),
    A.HorizontalFlip(p=0.4),
])


def augment_and_save_images():
    labels_to_augment = {0: 1, 2: 2}  # 对应的扩增倍数(初版，到submission20)
    labels_to_augment = {0: 3, 1: 1, 2: 5}  # 对应的扩增倍数（S21后）
    csv_file = 'D:/AI_Data/data_2/df_task2_train_challenge.csv'
    # csv_file = 'train_temp.csv'
    root_dir = 'D:/AI_Data/data_2/train'
    output_dir = 'D:/AI_Data/data2_aug2/augmented_train'
    output_csv = 'D:/AI_Data/data2_aug2/augmented_train.csv'

    data_frame = pd.read_csv(csv_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    new_data = []

    for idx, row in data_frame.iterrows():
        label = row['label']
        img_path = os.path.join(root_dir, row['image'])
        if os.path.isfile(img_path):
            img = np.array(Image.open(img_path).convert('L'))

            if label in labels_to_augment:
                for i in range(labels_to_augment[label]):
                    augmented = augmentation(image=img)
                    aug_image = augmented['image']

                    aug_image = aug_image.astype(np.uint8)
                    aug_img_name = f"{os.path.splitext(row['image'])[0]}_aug_{i}.png"
                    aug_LOCALIZER = f"{os.path.splitext(row['LOCALIZER'])[0]}_aug_{i}{os.path.splitext(row['LOCALIZER'])[1]}"

                    Image.fromarray(aug_image).save(os.path.join(output_dir, aug_img_name))

                    new_data.append({
                        'id_patient': row['id_patient'],
                        'side_eye': row['side_eye'],
                        'BScan': row['BScan'],
                        'image': aug_img_name,
                        'split_type': row['split_type'],
                        'label': row['label'],
                        'LOCALIZER': aug_LOCALIZER,
                        'sex': row['sex'],
                        'age': row['age'],
                        'num_current_visit': row['num_current_visit'],
                        'case': row['case']
                    })

            # 保存标签为1和未增强标签0和2的原始图像
            original_img_name = os.path.basename(row['image'])
            new_img_path = os.path.join(output_dir, original_img_name)
            Image.fromarray(img).save(new_img_path)
            row['image'] = original_img_name
            new_data.append(row.to_dict())

    # 保存新数据集
    new_data_frame = pd.DataFrame(new_data)
    new_data_frame.to_csv(output_csv, index=False)

    print("数据增强完成并保存到新的文件夹和文件中。")