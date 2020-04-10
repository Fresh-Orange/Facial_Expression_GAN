from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from torch.utils.data.sampler import Sampler


class FaceDataset(data.Dataset):
    def __init__(self, face_dir, transform, mode, config):
        """Initialize and preprocess the CelebA dataset."""
        self.face_dir = face_dir
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
                sub_list = os.listdir(full_sub_dir)
                sub_list = [sub for sub in sub_list if sub.endswith("full.jpg")]
                sub_list.sort(key=lambda x:int(x[:-9]))
                for file in sub_list:
                    full_file = os.path.join(full_sub_dir, file)
                    if "full" in file and os.path.isfile(full_file) and os.path.getsize(full_file) > 1024:
                        print("file", file)
                        self.train_dataset.append(os.path.join(sub_dir, file))
                        start_index = start_index + 1

        class_start_indices.append(start_index)  # 结尾的index
        self.class_start_indices = class_start_indices


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        fullface = dataset[index]
        face_id = fullface.split("_")[0]
        full_face = Image.open(os.path.join(self.face_dir, face_id+"_full.jpg"))
        crop_face = Image.open(os.path.join(self.face_dir, face_id+"_crop.jpg"))
        mask_face = Image.open(os.path.join(self.face_dir, face_id+"_mask.jpg"))
        mask_point_face = Image.open(os.path.join(self.face_dir, face_id + "_mask_points.jpg"))
        knockout_face = Image.open(os.path.join(self.face_dir, face_id+"_knockout.jpg"))

        return self.transform(full_face), self.transform(crop_face),self.transform(mask_face),self.transform(mask_point_face), self.transform(knockout_face)

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
        start_list = []
        for (a, b) in indices_pairs:
            start_index = random.randint(a,b-6)
            start_list.append(start_index)

        print([idx for a in start_list for idx in range(a, a+self.batch_size)])

        return (idx for a in start_list for idx in range(a, a+self.batch_size))

    def __len__(self):
        return len(self.indices)*self.batch_size

def get_loader(face_dir, config, image_size=(720, 544),
               batch_size=5, dataset='CelebA', mode='train', num_workers=1):
    """

    :param face_dir:
    :param config:
    :param image_size:
    :param batch_size: default 200, 意思是对于每个人每次都顺序取出200张
    :param dataset:
    :param mode:
    :param num_workers:
    :return:
    """
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = FaceDataset(face_dir,  transform, mode, config)
    subsetSampler = SubsetRandomSampler(dataset.class_start_indices, batch_size)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  #shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  sampler=subsetSampler)
    return data_loader