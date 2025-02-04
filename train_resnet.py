import argparse
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from model.resnet import load_resnet_18
from resnet import load_resnet_18
# from dataset.disfa_face import DISFA_Image_Face
from disfa_face import DISFA_Image_Face
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser("Resnet pretraining")
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--ind", type=int, default=0)

args = parser.parse_args()


device = 'cuda'
batch_size = args.batch_size
num_workers = args.num_workers


ALL_SUBJECTS = [['SN001', 'SN002', 'SN003', 'SN004', 'SN005'], ['SN006', 'SN007', 'SN008',
                'SN009', 'SN010'], ['SN011', 'SN012', 'SN013', 'SN016', 'SN017'], ['SN018',
                'SN021', 'SN023', 'SN024', 'SN025', 'SN026'], ['SN027', 'SN028', 'SN029',
                'SN030', 'SN031', 'SN032']]



def select_fold(ind):
    assert 0 <= ind < len(ALL_SUBJECTS)
    test_id = ALL_SUBJECTS[ind]
    train_id = [item for idx, sublist in enumerate(ALL_SUBJECTS) if idx != ind for item in sublist]
    return train_id, test_id


def test(model, test_dataloader, criterion):
    model.eval()  # 将模型设置为训练模式
    with torch.no_grad():
        total_loss = 0
        for img, target in test_dataloader:
            img = img.to(device)
            target = target.to(device)
            output = model(img) # 前向传播
            loss = criterion(output, target)  # 计算损失
            total_loss += loss.item()

    print(f'Test Loss: {total_loss / len(test_dataloader)}')
    logging.info(f'Test Loss: {total_loss / len(test_dataloader)}')
    model.train()
    return total_loss / len(test_dataloader)


def train(model, train_dataloader, test_dataloader, criterion, optimizer, epochs):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training started at {current_time}")
    model.train()  # 将模型设置为训练模式
    for epoch in range(epochs):
        total_loss = 0
        for i, (img, target) in enumerate(train_dataloader):
            img = img.to(device)
            target = target.to(device)
            optimizer.zero_grad()  # 清除所有优化的梯度
            output = model(img) # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()
            # print(i)
            # if i % 100 == 0:
            #     print(f'Epoch {epoch + 1}/{epochs}, Batch {i}/{len(train_dataloader)}, Loss: {loss.item():.4f} ')
            #     test(model, test_dataloader, criterion)

        test_loss = test(model, test_dataloader, criterion)
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}')
        logging.info(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}')
        writer.add_scalar('Training loss', total_loss / len(train_dataloader), epoch)
        writer.add_scalar('Test loss', test_loss, epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'resnet_' + str(epoch) + '.pth'))


    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training ended at {current_time}")
    return model

if __name__ == '__main__':
    train_fold, test_fold = select_fold(args.ind)


    writer = SummaryWriter('runs/fold_' + str(args.ind))
    save_path = './my_scripts/trained_resnet_with_fold_' + str(args.ind)
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_path, 'loss.log'), level=logging.INFO,
                        filemode='w', format='%(asctime)s:%(levelname)s:%(message)s')

    train_dataset = DISFA_Image_Face(
                        # root_dir='/media/mengting/Expansion/image_datasets/disfa/tmp/all',
                        root_dir = './all',
                        fold_subjects=train_fold,
                        split='train'
    )

    test_dataset = DISFA_Image_Face(
                            # root_dir='/media/mengting/Expansion/image_datasets/disfa/tmp/all',
                            root_dir = './all',
                            fold_subjects=test_fold, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    resnet = load_resnet_18()


    # freeze some parameters
    for name, param in resnet.named_parameters():
        if 'layer4' in name or 'linear' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # for name, param in resnet.named_parameters():
    #     print(name, param.requires_grad)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1.0e-3, betas=(0.9, 0.95), eps=1e-8)
    criterion = nn.MSELoss().to(device)
    resnet = resnet.to(device)
    print(len(train_loader), len(test_loader))
    
    model = train(resnet, train_loader, test_loader, criterion, optimizer, epochs=args.epochs)

    print('done')


