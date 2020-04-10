import os
import argparse
from data_loder import data_loader_augment_color_for_transfer as data_loader
from torch.backends import cudnn
import torch
from model.model import FeatureExtractNet, SNResIdDiscriminator
from model.model import NoTrackExpressionGenerater as ExpressionGenerater
from torchvision.utils import save_image
import sys
from utils.Logger import Logger
from torch.nn.functional import softplus
from pytorch_msssim import ssim


device = None

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


def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def main(config):
    # For fast training.
    cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if len(sys.argv) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    version = config.version
    beta1= 0.5
    beta2 = 0.999

    loader = data_loader.get_loader("/media/data2/laixc/AI_DATA/expression_transfer/face12/crop_face",
                                         "/media/data2/laixc/AI_DATA/expression_transfer/face12/points_face", config)
    G = ExpressionGenerater()
    FEN = FeatureExtractNet()
    color_D = SNResIdDiscriminator()

    #######   载入预训练网络   ######
    resume_iter = config.resume_iter
    ckpt_dir = "/media/data2/laixc/Facial_Expression_GAN/ckpt-{}".format(version)
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
    log = Logger(os.path.join(ckpt_dir,
                              'log.txt'))
    if os.path.exists(os.path.join(ckpt_dir,
                                  '{}-G.ckpt'.format(resume_iter))):
        G_path = os.path.join(ckpt_dir,
                              '{}-G.ckpt'.format(resume_iter))
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        IdD_path = os.path.join(ckpt_dir,
                              '{}-idD.ckpt'.format(resume_iter))
        color_D.load_state_dict(torch.load(IdD_path, map_location=lambda storage, loc: storage))

    else:
        resume_iter = 0


    #####  训练face2keypoint   ####
    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.9))
    idD_optimizer = torch.optim.Adam(color_D.parameters(), lr=0.001, betas=(0.5, 0.9))
    G.to(device)
    color_D.to(device)
    FEN.to(device)

    FEN.eval()

    log.print(config)

    # Start training from scratch or resume training.
    start_iters = resume_iter
    trigger_rec = 1
    data_iter = iter(loader)

    # Start training.
    print('Start training...')
    for i in range(start_iters, 150000):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        #faces, origin_points = next(data_iter)
        #_, target_points = next(data_iter)
        try:
            color_faces, faces = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            color_faces, faces = next(data_iter)
        rand_idx = torch.randperm(faces.size(0))
        condition_faces = faces[rand_idx]

        faces = faces.to(device)
        color_faces = color_faces.to(device)
        condition_faces = condition_faces.to(device)

        # =================================================================================== #
        #                               3. Train the discriminator                            #
        # =================================================================================== #

        # ID Dis
        id_real_loss = torch.mean(softplus(-color_D(faces, condition_faces)))  # big for real
        faces_fake = G(color_faces, condition_faces)
        id_fake_loss = torch.mean(softplus(color_D(faces_fake, condition_faces)))  # small for fake

        id_Dis_loss = id_real_loss + id_fake_loss

        idD_optimizer.zero_grad()
        id_Dis_loss.backward()
        idD_optimizer.step()



        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #

        n_critic = 1
        if (i + 1) % n_critic == 0:
            # Original-to-target domain.
            faces_fake = G(color_faces, condition_faces)

            g_id_loss = torch.mean(softplus(-color_D(faces_fake, condition_faces)))

            l1_loss = torch.mean(torch.abs(faces_fake - faces)) + (1 - ssim(faces_fake, faces))

            feature_loss = torch.mean(torch.abs(FEN(faces_fake) - FEN(faces)))

            lambda_l1 = config.lambda_l1
            lambda_id = config.lambda_id
            lambda_feature = config.lambda_feature
            g_loss =  lambda_id * g_id_loss + lambda_l1 * l1_loss + lambda_feature*feature_loss

            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            # Print out training information.
            if (i + 1) % 4 == 0:
                log.print("iter {} - id_real_loss {:.2}, "
                      "id_fake_loss {:.2} ,  g_id_loss {:.2}, L1_loss {:.2}, feature_loss {:.2}".format(i, id_real_loss.item(), id_fake_loss.item()
                                                                ,lambda_id * g_id_loss.item(), lambda_l1 * l1_loss,
                                                                                                                    lambda_feature *feature_loss.item()))

            sample_dir = "gan-sample-{}".format(version)
            if not os.path.isdir(sample_dir):
                os.mkdir(sample_dir)
            if (i + 1) % 24 == 0:
                with torch.no_grad():
                    fake_face = faces_fake[0]
                    condition_face = condition_faces[0]
                    color_face = color_faces[0]
                    #reconstruct = reconstructs[0]

                    sample_path_face = os.path.join(sample_dir, '{}-image-face.jpg'.format(i + 1))
                    save_image(denorm(condition_face.data.cpu()), sample_path_face)

                    sample_path_rec = os.path.join(sample_dir, '{}-image-color.jpg'.format(i + 1))
                    save_image(denorm(color_face.data.cpu()), sample_path_rec)

                    sample_path_fake = os.path.join(sample_dir, '{}-image-fake.jpg'.format(i + 1))
                    save_image(denorm(fake_face.data.cpu()), sample_path_fake)


                    print('Saved real and fake images into {}...'.format(sample_path_face))


        # Save model checkpoints.
        model_save_dir = "ckpt-{}".format(version)

        if (i + 1) % 1000 == 0:
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)
            G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(i + 1))
            torch.save(G.state_dict(), G_path)
            idD_path = os.path.join(model_save_dir, '{}-idD.ckpt'.format(i + 1))
            torch.save(color_D.state_dict(), idD_path)
            print('Saved model checkpoints into {}...'.format(model_save_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--resume_iter', type=int, default=14000)
    parser.add_argument('--version', type=str, default="256-level1")
    parser.add_argument('--dataset_level', '--list', nargs='+',
                        default=['easy', 'middle', 'hard'])
    parser.add_argument('--gpu', type=str, default="2")

    parser.add_argument('--lambda_l1', type=float, default=4)
    parser.add_argument('--lambda_id', type=float, default=0.1)
    parser.add_argument('--lambda_feature', type=float, default=2)

    config = parser.parse_args()
    main(config)
