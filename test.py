import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# from dataset.disfa_face import DISFA_Image_Face
# from model.resnet import load_resnet_18
from disfa_face import DISFA_Image_Face
from resnet import load_resnet_18

device = 'cuda'

parser = argparse.ArgumentParser("Resnet pretraining")
parser.add_argument("--ind", type=int, default=0)
args = parser.parse_args()

ALL_SUBJECTS = [['SN001', 'SN002', 'SN003', 'SN004', 'SN005'], ['SN006', 'SN007', 'SN008',
                'SN009', 'SN010'], ['SN011', 'SN012', 'SN013', 'SN016', 'SN017'], ['SN018',
                'SN021', 'SN023', 'SN024', 'SN025', 'SN026'], ['SN027', 'SN028', 'SN029',
                'SN030', 'SN031', 'SN032']]


def select_fold(ind):
    assert 0 <= ind < len(ALL_SUBJECTS)
    test_id = ALL_SUBJECTS[ind]
    train_id = [item for idx, sublist in enumerate(ALL_SUBJECTS) if idx != ind for item in sublist]
    return train_id, test_id


if __name__ == '__main__':
    #test_one_image()
    train_fold, test_fold = select_fold(args.ind)
    
    resnet = load_resnet_18()
    resnet.requires_grad_(False)

    # state = torch.load('/home/mengting/Desktop/my_scripts/trained_resnet_with_fold_' + str(args.ind) + '/resnet_20.pth')
    state = torch.load('./my_scripts/trained_resnet_with_fold_' + str(args.ind) + '/resnet_20.pth')
    to_tensor_test = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])

    resnet.load_state_dict(state)
    resnet = resnet.to(device)

    test_dataset = DISFA_Image_Face(
        # root_dir='/media/mengting/Expansion/image_datasets/disfa/tmp/all',
        root_dir='./all',
                            fold_subjects=test_fold, split='test')

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=1,
                            pin_memory=True, drop_last=False)

    with torch.no_grad():
        total_loss = 0
        for img, target in test_loader:
            img = img.to(device)
            target = target.to(device) * 5.0
            output = resnet(img) * 5.0
            mse = F.mse_loss(output, target)
            total_loss += mse.item()
        print("Total Loss (scaled): ", total_loss / len(test_loader))
        print("Total Loss (unscaled): ", (total_loss / len(test_loader))/25)

    print('done')