from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from torch.utils.data.sampler import Sampler


class FaceDataset(data.Dataset):
    def __init__(self, face_dir, keypoints_dir, transform, mode, config):
        """Initialize and preprocess the CelebA dataset."""
        self.face_dir = face_dir
        self.keypoints_dir = keypoints_dir
        self.transform = transform
        self.mode = mode
        self.config = config
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.class_start_indices = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        class_start_indices = []
        start_index = 0
        for sub_dir in os.listdir(self.face_dir):
            if int(sub_dir) > 12000 and "hard" not in self.config.dataset_level:
                continue
            if 12000 > int(sub_dir) > 11000 and "middle" not in self.config.dataset_level:
                continue
            if 11000 > int(sub_dir) > 10000 and "easy" not in self.config.dataset_level:
                continue
            print("Reading {}".format(sub_dir))
            full_sub_dir = os.path.join(self.face_dir, sub_dir)
            class_start_indices.append(start_index)
            if os.path.isdir(full_sub_dir):
                for file in os.listdir(full_sub_dir):
                    full_file = os.path.join(full_sub_dir, file)
                    if os.path.isfile(full_file) and os.path.getsize(full_file) > 1024:
                        self.train_dataset.append([os.path.join(sub_dir, file),
                                                   os.path.join(sub_dir, file)])
                        start_index = start_index + 1

        class_start_indices.append(start_index)  # 结尾的index
        self.class_start_indices = class_start_indices


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


class SubsetRandomSampler(Sampler):
    r"""
    by laixiancheng
    Arguments:
        indices (sequence): a sequence of start indices, eg. [0, 6, 14, 21]
        batch_size: batch_size for batch sampler
    """

    def __init__(self, indices, batch_size):
        self.indices = indices
        self.batch_size = batch_size

    def __iter__(self):
        indices_pairs = [(self.indices[i], self.indices[i+1]) for i in range(len(self.indices)-1)]

        return (idx for (a, b) in indices_pairs for idx in random.sample(range(a, b), self.batch_size))

    def __len__(self):
        return len(self.indices)*self.batch_size

def get_loader(face_dir, keypoint_dir, config, image_size=(224, 224),
               batch_size=8, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = FaceDataset(face_dir, keypoint_dir, transform, mode, config)
    subsetSampler = SubsetRandomSampler(dataset.class_start_indices, batch_size)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  #shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  sampler=subsetSampler)
    return data_loader