import os
import argparse
from data_loder import data_loader_augment as data_loader
from torch.backends import cudnn
import torch
from model.model import LandMarksDetect, SNResRealFakeDiscriminator, BigConvExpressionGenerater, FeatureExtractNet, SNResIdDiscriminator
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
    points_G = LandMarksDetect()
    G = BigConvExpressionGenerater()
    D = SNResRealFakeDiscriminator()
    FEN = FeatureExtractNet()
    id_D = SNResIdDiscriminator()

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

        points_G_path = os.path.join(ckpt_dir,
                                     '{}-pG.ckpt'.format(resume_iter))
        points_G.load_state_dict(torch.load(points_G_path, map_location=lambda storage, loc: storage))
    else:
        resume_iter = 0


    #####  训练face2keypoint   ####
    points_G_optimizer = torch.optim.Adam(points_G.parameters(), lr=0.0001, betas=(0.5, 0.9))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.9))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.9))
    idD_optimizer = torch.optim.Adam(id_D.parameters(), lr=0.001, betas=(0.5, 0.9))
    G.to(device)
    id_D.to(device)
    D.to(device)
    points_G.to(device)
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
            rotate_faces, faces, origin_points = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            rotate_faces, faces, origin_points = next(data_iter)

        rand_idx = torch.randperm(origin_points.size(0))
        target_points = origin_points[rand_idx]
        target_faces = faces[rand_idx]

        faces = faces.to(device)
        rotate_faces = rotate_faces.to(device)
        target_faces = target_faces.to(device)
        origin_points = origin_points.to(device)
        target_points = target_points.to(device)

        # =================================================================================== #
        #                               3. Train the discriminator                            #
        # =================================================================================== #

        # Real fake Dis
        real_loss = torch.mean(softplus(-D(faces)))  # big for real
        faces_fake = G(rotate_faces, target_points)
        fake_loss = torch.mean(softplus(D(faces_fake)))  # small for fake

        Dis_loss = real_loss + fake_loss

        D_optimizer.zero_grad()
        Dis_loss.backward()
        D_optimizer.step()

        # ID Dis
        id_real_loss = torch.mean(softplus(-id_D(faces, target_faces)))  # big for real
        faces_fake = G(rotate_faces, target_points)
        id_fake_loss = torch.mean(softplus(id_D(faces, faces_fake)))  # small for fake

        id_Dis_loss = id_real_loss + id_fake_loss

        idD_optimizer.zero_grad()
        id_Dis_loss.backward()
        idD_optimizer.step()


        # if (i + 1) % 5 == 0:
        #     print("iter {} - d_real_loss {:.2}, d_fake_loss {:.2}, d_loss_gp {:.2}".format(i,real_loss.item(),
        #                                                                                              fake_loss.item(),
        #                                                                                              lambda_gp * d_loss_gp
        #                                                                                              ))

        # =================================================================================== #
        #                               3. Train the keypointsDetecter                        #
        # =================================================================================== #

        points_detect = points_G(faces)
        detecter_loss_clear = torch.mean(torch.abs(points_detect - origin_points))
        # faces_fake = G(faces, target_points)
        # reconstructs = G(faces_fake, origin_points)
        # detecter_loss_vague = torch.mean(torch.abs(points_G(reconstructs) - origin_points))
        #
        # lambda_vague = 0.1
        # detecter_loss = detecter_loss_clear + lambda_vague*detecter_loss_vague
        detecter_loss = detecter_loss_clear
        points_G_optimizer.zero_grad()
        detecter_loss.backward()
        points_G_optimizer.step()

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #

        n_critic = 1
        if (i + 1) % n_critic == 0:
            # Original-to-target domain.
            faces_fake = G(rotate_faces, target_points)
            predict_points = points_G(faces_fake)
            g_keypoints_loss = torch.mean(torch.abs(predict_points - target_points))

            g_fake_loss = torch.mean(softplus(-D(faces_fake)))

            # reconstructs = G(faces_fake, origin_points)
            # g_cycle_loss = torch.mean(torch.abs(reconstructs - faces))
            g_id_loss = torch.mean(softplus(-id_D(faces, faces_fake)))

            l1_loss = torch.mean(torch.abs(faces_fake - target_faces))

            feature_loss = torch.mean(torch.abs(FEN(faces_fake) - FEN(target_faces)))

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
                      + lambda_id * g_id_loss + lambda_l1 * l1_loss + lambda_feature*feature_loss

            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            # Print out training information.
            if (i + 1) % 4 == 0:
                log.print("iter {} - d_real_loss {:.2}, d_fake_loss {:.2}, id_real_loss {:.2}, "
                      "id_fake_loss {:.2} , g_keypoints_loss {:.2}, "
                      "g_fake_loss {:.2}, g_id_loss {:.2}, L1_loss {:.2}, feature_loss {:.2}".format(i, real_loss.item(), fake_loss.item()
                                                                                                     ,id_real_loss.item(), id_fake_loss.item()
                                                               , lambda_keypoint*g_keypoints_loss.item(), lambda_fake*g_fake_loss.item()
                                                                ,lambda_id * g_id_loss.item(), lambda_l1 * l1_loss,
                                                                                                                    lambda_feature *feature_loss.item()))

            sample_dir = "gan-sample-{}".format(version)
            if not os.path.isdir(sample_dir):
                os.mkdir(sample_dir)
            if (i + 1) % 24 == 0:
                with torch.no_grad():
                    target_point = target_points[0]
                    fake_face = faces_fake[0]
                    #face = faces[0]
                    rotate_face = rotate_faces[0]
                    #reconstruct = reconstructs[0]
                    predict_point = predict_points[0]

                    sample_path_face = os.path.join(sample_dir, '{}-image-face.jpg'.format(i + 1))
                    save_image(denorm(rotate_face.data.cpu()), sample_path_face)

                    # sample_path_rec = os.path.join(sample_dir, '{}-image-reconstruct.jpg'.format(i + 1))
                    # save_image(denorm(reconstruct.data.cpu()), sample_path_rec)

                    sample_path_fake = os.path.join(sample_dir, '{}-image-fake.jpg'.format(i + 1))
                    save_image(denorm(fake_face.data.cpu()), sample_path_fake)

                    sample_path_target = os.path.join(sample_dir, '{}-image-target_point.jpg'.format(i + 1))
                    save_image(denorm(target_point.data.cpu()), sample_path_target)

                    sample_path_predict_points = os.path.join(sample_dir, '{}-image-predict_point.jpg'.format(i + 1))
                    save_image(denorm(predict_point.data.cpu()), sample_path_predict_points)

                    print('Saved real and fake images into {}...'.format(sample_path_face))


        # Save model checkpoints.
        model_save_dir = "ckpt-{}".format(version)

        if (i + 1) % 1000 == 0:
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)
            point_G_path = os.path.join(model_save_dir, '{}-pG.ckpt'.format(i + 1))
            torch.save(points_G.state_dict(), point_G_path)
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
    parser.add_argument('--lambda_keypoint', type=float, default=100)
    parser.add_argument('--lambda_fake', type=float, default=0.1)
    parser.add_argument('--lambda_id', type=float, default=0.1)
    parser.add_argument('--lambda_feature', type=float, default=2)

    config = parser.parse_args()
    main(config)
