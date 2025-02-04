import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from disfa_face import DISFA_Image_Face
from resnet import load_resnet_18

device = 'cuda'

parser = argparse.ArgumentParser("Resnet pretraining")
parser.add_argument("--ind", type=int, default=0)
args = parser.parse_args()

ALL_SUBJECTS = [['SN001', 'SN002', 'SN003', 'SN004', 'SN005'], 
                 ['SN006', 'SN007', 'SN008', 'SN009', 'SN010'], 
                 ['SN011', 'SN012', 'SN013', 'SN016', 'SN017'], 
                 ['SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026'], 
                 ['SN027', 'SN028', 'SN029', 'SN030', 'SN031', 'SN032']]

def select_fold(ind):
    assert 0 <= ind < len(ALL_SUBJECTS)
    test_id = ALL_SUBJECTS[ind]
    train_id = [item for idx, sublist in enumerate(ALL_SUBJECTS) if idx != ind for item in sublist]
    return train_id, test_id

if __name__ == '__main__':
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/evaluation')

    for fold in range(5):  # Loop over each fold
        train_fold, test_fold = select_fold(fold)
        
        for model_index in range(0, 50, 10):  # Loop over model versions (0, 10, 20, 30, 40)
            model_filename = f'./my_scripts/trained_resnet_with_fold_{fold}/resnet_{model_index}.pth'
            print(f"Testing fold {fold} with model {model_filename}")

            resnet = load_resnet_18()
            resnet.requires_grad_(False)

            # Load the model state
            state = torch.load(model_filename)
            resnet.load_state_dict(state)
            resnet = resnet.to(device)

            # Prepare the test dataset
            test_dataset = DISFA_Image_Face(
                root_dir='./all',
                fold_subjects=test_fold, 
                split='test'
            )

            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=1,
                                     pin_memory=True, drop_last=False)

            # Evaluate the model
            with torch.no_grad():
                total_loss = 0
                for img, target in test_loader:
                    img = img.to(device)
                    target = target.to(device) * 5.0  # Scale target
                    output = resnet(img) * 5.0  # Scale output
                    mse = F.mse_loss(output, target)
                    total_loss += mse.item()
                
                average_loss = total_loss / len(test_loader)
                print(f"Total Loss for fold {fold}, resnet_{model_index}: ", average_loss)

                # Log the average loss to TensorBoard
                writer.add_scalar(f'Loss/fold_{fold}/resnet_{model_index}', average_loss)

    # Close the TensorBoard writer
    writer.close()
    print('done')