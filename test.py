import os
import argparse
from solver import Solver
import data_loader
from torch.backends import cudnn
import torch
from model import LandMarksDetect
from torchvision.utils import save_image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def main():
    G_path = os.path.join("/media/data2/laixc/Facial_Expression_GAN/ckpt", '10000-G.ckpt')
    points_G = LandMarksDetect()
    points_G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    loader = data_loader.get_loader("/media/data2/laixc/AI_DATA/expression_transfer/face2/crop_face",
                                    "/media/data2/laixc/AI_DATA/expression_transfer/face2/points_face")
    loader = iter(loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    points_G.to(device)

    sample_dir = "validation_sample"
    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)

    with torch.no_grad():
        for i, (faces, points) in enumerate(loader):
            faces = faces.to(device)
            points_fake = points_G(faces)
            true_points = points[0]
            fake = points_fake[0]
            face = faces[0]
            sample_path_face = os.path.join(sample_dir, '{}-image-face.jpg'.format(i + 1))
            save_image(denorm(face.data.cpu()), sample_path_face)
            sample_path_real = os.path.join(sample_dir, '{}-image-real.jpg'.format(i + 1))
            save_image(denorm(true_points.data.cpu()), sample_path_real)
            sample_path_fake = os.path.join(sample_dir, '{}-image-fake.jpg'.format(i + 1))
            save_image(denorm(fake.data.cpu()), sample_path_fake)
            print('Saved real and fake images into {}...'.format(sample_path_real))


if __name__ == '__main__':
    main()