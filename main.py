import os
import argparse
from solver import Solver
import data_loader
from torch.backends import cudnn
import torch
from model import LandMarksDetect, RealFakeDiscriminator, ExpressionGenerater
from torchvision.utils import save_image
import math

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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

def main():
    # For fast training.
    cudnn.benchmark = True

    loader = data_loader.get_loader("/media/data2/laixc/AI_DATA/expression_transfer/face1/crop_face",
                                         "/media/data2/laixc/AI_DATA/expression_transfer/face1/points_face")
    points_G = LandMarksDetect()
    G = ExpressionGenerater()
    D = RealFakeDiscriminator()

    #######   载入预训练网络   ######
    points_G_path = os.path.join("/media/data2/laixc/Facial_Expression_GAN/face2keypoint_ckpt", '{}-G.ckpt'.format(45000))
    points_G.load_state_dict(torch.load(points_G_path, map_location=lambda storage, loc: storage))
    resume_iter = 50000
    G_path = os.path.join("/media/data2/laixc/Facial_Expression_GAN/ckpt",
                                  '{}-G.ckpt'.format(resume_iter))
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    #####  训练face2keypoint   ####
    points_G_optimizer = torch.optim.Adam(points_G.parameters(), lr=0.0001)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    G.to(device)
    D.to(device)
    points_G.to(device)



    # Start training from scratch or resume training.
    start_iters = resume_iter

    # Start training.
    print('Start training...')
    for i in range(start_iters, 100000):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        data_iter = iter(loader)
        faces, origin_points = next(data_iter)
        _, target_points = next(data_iter)
        faces = faces.to(device)
        origin_points = origin_points.to(device)
        target_points = target_points.to(device)

        # =================================================================================== #
        #                               3. Train the discriminator                            #
        # =================================================================================== #

        real_loss = - torch.mean(D(faces))  # 1 for real
        faces_fake = G(faces, target_points)
        fake_loss = torch.mean(D(faces_fake))  # 0 for fake

        # Compute loss for gradient penalty.
        alpha = torch.rand(faces.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * faces.data + (1 - alpha) * faces_fake.data).requires_grad_(True)
        out_src = D(x_hat)
        d_loss_gp = gradient_penalty(out_src, x_hat)

        lambda_gp = 10
        Dis_loss = real_loss + fake_loss + lambda_gp * d_loss_gp

        D_optimizer.zero_grad()
        Dis_loss.backward()
        D_optimizer.step()

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

        n_critic = 5
        if (i + 1) % n_critic == 0:
            # Original-to-target domain.
            faces_fake = G(faces, target_points)
            predict_points = points_G(faces_fake)
            g_keypoints_loss = torch.mean(torch.abs(predict_points - target_points))

            g_fake_loss = - torch.mean(D(faces_fake))

            reconstructs = G(faces_fake, origin_points)
            g_cycle_loss = torch.mean(torch.abs(reconstructs - faces))

            lambda_rec = 100*(1/(i+1))+10
            lambda_keypoint = 10*(i/25000)
            lambda_fake = min((i+100)/500000, 1)
            g_loss = lambda_keypoint * g_keypoints_loss + lambda_fake*g_fake_loss + lambda_rec * g_cycle_loss

            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            # Print out training information.
            if (i + 1) % 5 == 0:
                print("iter {} - d_real_loss {:.2}, d_fake_loss {:.2}, d_loss_gp {:.2} , g_keypoints_loss {:.2},"
                      "g_fake_loss {:.2}, g_cycle_loss {:.2}, detecter_loss {:.2}".format(i, real_loss.item(), fake_loss.item(), lambda_gp * d_loss_gp
                                                               , lambda_keypoint*g_keypoints_loss.item(), lambda_fake*g_fake_loss.item()
                                                               , lambda_rec*g_cycle_loss.item(), detecter_loss.item()))

            sample_dir = "gan-sample"
            if not os.path.isdir(sample_dir):
                os.mkdir(sample_dir)
            if (i + 1) % 50 == 0:
                with torch.no_grad():
                    target_point = target_points[0]
                    fake_face = faces_fake[0]
                    face = faces[0]
                    reconstruct = reconstructs[0]
                    predict_point = predict_points[0]

                    sample_path_face = os.path.join(sample_dir, '{}-image-face.jpg'.format(i + 1))
                    save_image(denorm(face.data.cpu()), sample_path_face)

                    sample_path_rec = os.path.join(sample_dir, '{}-image-reconstruct.jpg'.format(i + 1))
                    save_image(denorm(reconstruct.data.cpu()), sample_path_rec)

                    sample_path_fake = os.path.join(sample_dir, '{}-image-fake.jpg'.format(i + 1))
                    save_image(denorm(fake_face.data.cpu()), sample_path_fake)

                    sample_path_target = os.path.join(sample_dir, '{}-image-target_point.jpg'.format(i + 1))
                    save_image(denorm(target_point.data.cpu()), sample_path_target)

                    sample_path_predict_points = os.path.join(sample_dir, '{}-image-predict_point.jpg'.format(i + 1))
                    save_image(denorm(predict_point.data.cpu()), sample_path_predict_points)

                    print('Saved real and fake images into {}...'.format(sample_path_rec))


        # Save model checkpoints.
        model_save_dir = "ckpt"

        if (i + 1) % 1000 == 0:
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)
            point_G_path = os.path.join(model_save_dir, '{}-pG.ckpt'.format(i + 1))
            torch.save(points_G.state_dict(), point_G_path)
            G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(i + 1))
            torch.save(G.state_dict(), G_path)
            D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(i + 1))
            torch.save(D.state_dict(), D_path)
            print('Saved model checkpoints into {}...'.format(model_save_dir))



if __name__ == '__main__':
    main()