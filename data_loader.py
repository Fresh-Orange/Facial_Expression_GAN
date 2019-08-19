from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class FaceDataset(data.Dataset):
    def __init__(self, face_dir, keypoints_dir, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.face_dir = face_dir
        self.keypoints_dir = keypoints_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        for sub_dir in os.listdir(self.face_dir):
            print("Reading {}".format(sub_dir))
            full_sub_dir = os.path.join(self.face_dir, sub_dir)
            if os.path.isdir(full_sub_dir):
                for file in os.listdir(full_sub_dir):
                    full_file = os.path.join(full_sub_dir, file)
                    if os.path.isfile(full_file) and os.path.getsize(full_file) > 1024:
                        self.train_dataset.append([os.path.join(sub_dir, file),
                                                   os.path.join(sub_dir, file)])


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        facefile, keypointfile = dataset[index]
        faceimage = Image.open(os.path.join(self.face_dir, facefile))
        keypointimage = Image.open(os.path.join(self.keypoints_dir, keypointfile))
        return self.transform(faceimage), self.transform(keypointimage)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(face_dir, keypoint_dir, image_size=(116, 128),
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = FaceDataset(face_dir, keypoint_dir, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader