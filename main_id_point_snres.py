#
# 1. 使用了关键点判别器的版本，同时保留关键点检测器
# 2. 判别器使用SNRes版本
#
import os
import argparse
from data_loder import data_loader
from torch.backends import cudnn
import torch
from model.model import SNResRealFakeDiscriminator, ExpressionGenerater, SNResIdDiscriminator, SNResKeypointDiscriminator
from torchvision.utils import save_image
import sys
from utils.Logger import Logger
from torch.nn.functional import softplus

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
    D = SNResRealFakeDiscriminator()
    #FEN = FeatureExtractNet()
    id_D = SNResIdDiscriminator()
    kp_D = SNResKeypointDiscriminator()
    #points_G = LandMarksDetect()

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

        D_path = os.path.join(ckpt_dir,
                              '{}-D.ckpt'.format(resume_iter))
        D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

        IdD_path = os.path.join(ckpt_dir,
                              '{}-idD.ckpt'.format(resume_iter))
        id_D.load_state_dict(torch.load(IdD_path, map_location=lambda storage, loc: storage))

        kp_D_path = os.path.join(ckpt_dir,
                                     '{}-kpD.ckpt'.format(resume_iter))
        kp_D.load_state_dict(torch.load(kp_D_path, map_location=lambda storage, loc: storage))

        #points_G_path = os.path.join(ckpt_dir,
        #                            '{}-pG.ckpt'.format(resume_iter))
        #points_G.load_state_dict(torch.load(points_G_path, map_location=lambda storage, loc: storage))
    else:
        resume_iter = 0


    #####  训练face2keypoint   ####
    #points_G_optimizer = torch.optim.Adam(points_G.parameters(), lr=0.0001, betas=(0.5, 0.9))
    kp_D_optimizer = torch.optim.Adam(kp_D.parameters(), lr=0.0001, betas=(0.5, 0.9))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.9))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.9))
    idD_optimizer = torch.optim.Adam(id_D.parameters(), lr=0.001, betas=(0.5, 0.9))
    G.to(device)
    id_D.to(device)
    D.to(device)
    kp_D.to(device)
    #points_G.to(device)
    #FEN.to(device)

    #FEN.eval()

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
            faces, origin_points = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            faces, origin_points = next(data_iter)
        rand_idx = torch.randperm(origin_points.size(0))
        target_points = origin_points[rand_idx]
        target_faces = faces[rand_idx]

        faces = faces.to(device)
        target_faces = target_faces.to(device)
        origin_points = origin_points.to(device)
        target_points = target_points.to(device)

        # =================================================================================== #
        #                               3. Train the discriminator                            #
        # =================================================================================== #

        # Real fake Dis
        real_loss = torch.mean(softplus(-D(faces)))  # big for real
        faces_fake = G(faces, target_points)
        fake_loss = torch.mean(softplus(D(faces_fake)))  # small for fake


        Dis_loss = real_loss + fake_loss

        D_optimizer.zero_grad()
        Dis_loss.backward()
        D_optimizer.step()

        # ID Dis
        d_real = softplus(-id_D(faces, target_faces))
        #print("d_real", d_real)
        id_real_loss = torch.mean(d_real)  # big for real
        faces_fake = G(faces, target_points)
        d_fake = softplus(id_D(faces, faces_fake))
        #print("d_fake", d_fake)
        id_fake_loss = torch.mean(d_fake)  # small for fake


        id_Dis_loss = id_real_loss + id_fake_loss

        idD_optimizer.zero_grad()
        id_Dis_loss.backward()
        idD_optimizer.step()

        # Keypoints Dis
        kp_real_loss = torch.mean(softplus(-kp_D(target_faces, target_points)))  # big for real
        faces_fake = G(faces, target_points)
        kp_fake_loss = torch.mean(softplus(kp_D(faces_fake, target_points)))  # small for fake

        kp_lambda_gp = 10
        kp_Dis_loss = kp_real_loss + kp_fake_loss

        kp_D_optimizer.zero_grad()
        kp_Dis_loss.backward()
        kp_D_optimizer.step()


        # if (i + 1) % 5 == 0:
        #     print("iter {} - d_real_loss {:.2}, d_fake_loss {:.2}, d_loss_gp {:.2}".format(i,real_loss.item(),
        #                                                                                              fake_loss.item(),
        #                                                                                              lambda_gp * d_loss_gp
        #                                                                                              ))



        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #

        n_critic = 1
        if (i + 1) % n_critic == 0:
            # Original-to-target domain.
            faces_fake = G(faces, target_points)
            g_keypoints_loss = torch.mean(softplus(-kp_D(faces_fake, target_points)))

            g_fake_loss = torch.mean(softplus(-D(faces_fake)))

            # reconstructs = G(faces_fake, origin_points)
            # g_cycle_loss = torch.mean(torch.abs(reconstructs - faces))
            g_id_loss = torch.mean(softplus(-id_D(faces, faces_fake)))

            l1_loss = torch.mean(torch.abs(faces_fake - target_faces))

            #feature_loss = torch.mean(torch.abs(FEN(faces_fake) - FEN(target_faces)))

            # 轮流训练
            # if (i+1) % 50 == 0:
            #     trigger_rec = 1 - trigger_rec
            #     print("trigger_rec : ", trigger_rec)
            lambda_rec = config.lambda_rec  # 2 to 4 to 8
            lambda_l1 = config.lambda_l1
            lambda_keypoint = config.lambda_keypoint   # 100 to 50
            lambda_fake = config.lambda_fake
            lambda_id = config.lambda_id
            lambda_feature = config.lambda_feature
            g_loss = lambda_keypoint * g_keypoints_loss + lambda_fake*g_fake_loss \
                      + lambda_id * g_id_loss + lambda_l1 * l1_loss# + lambda_feature*feature_loss

            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            # Print out training information.
            if (i + 1) % 4 == 0:
                log.print("iter {} - d_real_loss {:.2}, d_fake_loss {:.2}, id_real_loss {:.2}, "
                      "id_fake_loss {:.2} , g_keypoints_loss {:.2}, "
                      "g_fake_loss {:.2}, g_id_loss {:.2}, L1_loss {:.2}".format(i, real_loss.item(), fake_loss.item()
                                                                                                     ,id_real_loss.item(), id_fake_loss.item()
                                                               , lambda_keypoint*g_keypoints_loss.item(), lambda_fake*g_fake_loss.item()
                                                                ,lambda_id * g_id_loss.item(), lambda_l1 * l1_loss))

            sample_dir = "gan-sample-{}".format(version)
            if not os.path.isdir(sample_dir):
                os.mkdir(sample_dir)
            if (i + 1) % 24 == 0:
                with torch.no_grad():
                    target_point = target_points[0]
                    fake_face = faces_fake[0]
                    face = faces[0]
                    #reconstruct = reconstructs[0]

                    sample_path_face = os.path.join(sample_dir, '{}-image-face.jpg'.format(i + 1))
                    save_image(denorm(face.data.cpu()), sample_path_face)

                    # sample_path_rec = os.path.join(sample_dir, '{}-image-reconstruct.jpg'.format(i + 1))
                    # save_image(denorm(reconstruct.data.cpu()), sample_path_rec)

                    sample_path_fake = os.path.join(sample_dir, '{}-image-fake.jpg'.format(i + 1))
                    save_image(denorm(fake_face.data.cpu()), sample_path_fake)

                    sample_path_target = os.path.join(sample_dir, '{}-image-target_point.jpg'.format(i + 1))
                    save_image(denorm(target_point.data.cpu()), sample_path_target)


                    print('Saved real and fake images into {}...'.format(sample_path_face))


        # Save model checkpoints.
        model_save_dir = "ckpt-{}".format(version)

        if (i + 1) % 1000 == 0:
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)
            kp_D_path = os.path.join(model_save_dir, '{}-kpD.ckpt'.format(i + 1))
            torch.save(kp_D.state_dict(), kp_D_path)
            G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(i + 1))
            torch.save(G.state_dict(), G_path)
            D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(i + 1))
            torch.save(D.state_dict(), D_path)
            idD_path = os.path.join(model_save_dir, '{}-idD.ckpt'.format(i + 1))
            torch.save(id_D.state_dict(), idD_path)
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
    parser.add_argument('--lambda_rec', type=float, default=8)
    parser.add_argument('--lambda_keypoint', type=float, default=0.1)
    parser.add_argument('--lambda_fake', type=float, default=0.1)
    parser.add_argument('--lambda_id', type=float, default=0.1)
    parser.add_argument('--lambda_feature', type=float, default=2)

    config = parser.parse_args()
    main(config)
