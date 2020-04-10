import os
from data_loder import data_loader
from torch.backends import cudnn
import torch
from model.model import LandMarksDetect
from torchvision.utils import save_image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"




def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def to_image(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    x = denorm(x.data.cpu())
    ndarr = x.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = ndarr
    return im

def main():
    # For fast training.
    cudnn.benchmark = True

    loader = data_loader.get_loader("/media/data2/laixc/AI_DATA/expression_transfer/face1/crop_face",
                                         "/media/data2/laixc/AI_DATA/expression_transfer/face1/points_face")
    points_G = LandMarksDetect()
    resume_iter = 45000
    G_path = os.path.join("/media/data2/laixc/Facial_Expression_GAN/face2keypoint_ckpt", '{}-G.ckpt'.format(resume_iter))
    points_G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    optimizer = torch.optim.Adam(points_G.parameters(), lr=0.0001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points_G.to(device)



    # Start training from scratch or resume training.
    start_iters = resume_iter

    # Start training.
    print('Start training...')
    for i in range(start_iters, 50000):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        data_iter = iter(loader)
        faces, points = next(data_iter)
        #print("face size {}".format(faces.size()))
        #print("point size {}".format(points.size()))
        faces = faces.to(device)
        points = points.to(device)

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #

        # Original-to-target domain.
        points_fake = points_G(faces)
        L1_loss = torch.mean(torch.abs(points_fake - points))


        optimizer.zero_grad()
        L1_loss.backward()
        optimizer.step()



        # Print out training information.
        if (i + 1) % 5 == 0:
            print("iter {} - loss {}".format(i, L1_loss.item()))

        sample_dir = "sample"
        if not os.path.isdir(sample_dir):
            os.mkdir(sample_dir)
        if (i + 1) % 20 == 0:
            with torch.no_grad():
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

        # Save model checkpoints.
        model_save_dir = "f2k_ckpt"

        if (i + 1) % 1000 == 0:
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)
            G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(i + 1))
            torch.save(points_G.state_dict(), G_path)
            print('Saved model checkpoints into {}...'.format(model_save_dir))



if __name__ == '__main__':
    main()