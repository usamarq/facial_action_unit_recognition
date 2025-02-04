import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# EXCEL_ROOT = '/media/mengting/Expansion/image_datasets/disfa/DISFA_Processed/processed_labels_ice'
EXCEL_ROOT = './processed_labels_ice'


to_tensor_train = transforms.Compose([
 	transforms.RandomHorizontalFlip(p=0.05),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomRotation(5),
	 transforms.ToTensor(),
])

to_tensor_test = transforms.ToTensor()




class DISFA_Image_Face(Dataset):
    # 16 2 16 2 0.9 fasking None
    def __init__(self,
        root_dir: str,
        fold_subjects,
        split
    ):
        self.img_size = 224
        self.root_dir = root_dir
        self.fold_subjects = fold_subjects
        self.split = split

        self.metadata = self._get_image_list()

    def __getitem__(self, index):
        meta = self.metadata[index]

        if self.split == 'train':
            img = Image.open(os.path.join(meta))
            img = to_tensor_train(img)
        else:
            img = Image.open(os.path.join(meta))
            img = to_tensor_test(img)

        AU_labels = self._retrive_AUs(meta)

        return img, torch.from_numpy(AU_labels).float()


    def __len__(self) -> int:
        return len(self.metadata)

    def _retrive_AUs(self, meta):
        sub = meta.split('\\')[-2]
        csv_path = os.path.join(EXCEL_ROOT, sub + '.csv')
        df = pd.read_csv(csv_path)
        AU_list = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

        img_name = meta.split('\\')[-1]
        frame = df[df['Frames'] == int(img_name.replace('.jpg', '')) + 1]
        tt = []
        for i in AU_list:
            au_cur = 'au' + str(i)
            value = frame[au_cur].values[0]
            tt.append(value)

        return np.array(tt) / 5.0

    def _get_image_list(self):
        subjects = self.fold_subjects

        res = []
        for subject in subjects:
            cur_subject_path = os.path.join(os.path.join(self.root_dir, subject))
            images = os.listdir(cur_subject_path)
            for image in images:
                cur_image_path = os.path.join(cur_subject_path, image)
                res.append(cur_image_path)
        return res



if __name__ == '__main__':
    train_dataset = DISFA_Image_Face(
        # root_dir='/media/mengting/Expansion/image_datasets/disfa/tmp/all',
        root_dir = './all',
        fold_subjects=['SN001', 'SN002', 'SN003', 'SN004', 'SN005'], split='test'
    )

    print('done')
