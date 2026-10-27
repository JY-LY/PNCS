# https://github.com/eddiexia98/PaPi/blob/main/utils/fmnist.py

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets


from .utils_algo import generate_uniform_cv_candidate_labels
from .cutout import Cutout


def load_fmnist(args):
    
    test_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])
    
    original_train = dsets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    ori_data, ori_labels = original_train.data, original_train.targets.long()
    class_name = original_train.classes
    test_dataset = dsets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size*4, \
        shuffle=False, num_workers=args.workers, \
        sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    )
    

    partialY_matrix = generate_uniform_cv_candidate_labels(args, ori_labels)
        
    
    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('Partial labels correctly loaded !')
    else:
        print('Inconsistent permutation !')

    print('Average candidate num: ', partialY_matrix.sum(1).mean())
    
    partial_training_dataset = FMNIST_Partialize(ori_data, partialY_matrix.float(), ori_labels.float())
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_training_dataset)
    
    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    return partial_training_dataloader, partialY_matrix, train_sampler, test_loader, class_name


class FMNIST_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, 4, padding_mode='reflect'),
            transforms.Grayscale(3), # 3
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ])
        
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.ToPILImage(),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        
        each_image1 = self.transform1(self.ori_images[index])
        each_image2 = self.transform2(self.ori_images[index])
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image1, each_image2, each_label, each_true_label, index

