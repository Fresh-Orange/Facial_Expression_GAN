import os
import argparse
from data_loder import data_loader_fusion
from torch.backends import cudnn
import torch
from model.model import RealFakeDiscriminator, FusionGenerater
from torchvision.utils import save_image
import sys
from discarded.with_mobilenet import PoseEstimationWithMobileNet
import collections


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

def load_state(net, checkpoint):
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)

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

    loader = data_loader_fusion.get_loader("/media/data2/laixc/AI_DATA/expression_transfer/face12/fusion_face", config)
    G = FusionGenerater()
    D = RealFakeDiscriminator()
    # FEN = FeatureExtractNet()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("/media/data2/laixc/Facial_Expression_GAN/light-human-pose/files/checkpoint_iter_370000.pth", map_location="cpu")
    load_state(net, checkpoint)
    net = net.eval()
    posenet = net.cuda()

    #######   载入预训练网络   ######
    resume_iter = config.resume_iter
    ckpt_dir = "/media/data2/laixc/Facial_Expression_GAN/fusion-ckpt-{}".format(version)
    if os.path.exists(os.path.join(ckpt_dir,
                                  '{}-G.ckpt'.format(resume_iter))):
        G_path = os.path.join(ckpt_dir,
                              '{}-G.ckpt'.format(resume_iter))
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        D_path = os.path.join(ckpt_dir,
                              '{}-D.ckpt'.format(resume_iter))
        D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        print("Load ckpt !!!")
    else:
        print("Found NO ckpt !!!")
        resume_iter = 0


    #####  训练face2keypoint   ####
    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.9))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.9))
    G.to(device)
    D.to(device)
    # FEN.to(device)

    # FEN.eval()



    # Start training from scratch or resume training.
    start_iters = resume_iter
    data_iter = iter(loader)

    # Start training.
    print('Start training...')
    for i in range(start_iters, 100000):
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        try:
            fullfaces, cropfaces, maskfaces, knockoutfaces = next(data_iter)
        except StopIteration:
            print("DATA STOPPED !!")
            data_iter = iter(loader)
            fullfaces, cropfaces, maskfaces, knockoutfaces = next(data_iter)

        rand_idx = torch.randperm(fullfaces.size(0))
        target_crops = cropfaces[rand_idx]
        target_masks = maskfaces[rand_idx]
        target_fulls = fullfaces[rand_idx]

        fullfaces = fullfaces.to(device)    # TODO: alt. change to knockout
        target_crops = target_crops.to(device)
        target_masks = target_masks.to(device)
        target_fulls = target_fulls.to(device)

        # =================================================================================== #
        #                               3. Train the discriminator                            #
        # =================================================================================== #

        # Real fake Dis
        real_loss = - torch.mean(D(fullfaces))  # big for real
        faces_fake = G(fullfaces, target_masks, target_crops)
        fake_loss = torch.mean(D(faces_fake))  # small for fake

        # Compute loss for gradient penalty.
        alpha = torch.rand(fullfaces.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * fullfaces.data + (1 - alpha) * faces_fake.data).requires_grad_(True)
        out_src = D(x_hat)
        d_loss_gp = gradient_penalty(out_src, x_hat)

        lambda_gp = 10
        Dis_loss = real_loss + fake_loss + lambda_gp * d_loss_gp

        D_optimizer.zero_grad()
        Dis_loss.backward()
        D_optimizer.step()


        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #

        n_critic = 4
        if (i + 1) % n_critic == 0:
            # Original-to-target domain.
            faces_fake = G(fullfaces, target_masks, target_crops)

            g_fake_loss = - torch.mean(D(faces_fake))

            # reconstructs = G(faces_fake, origin_points)
            # g_cycle_loss = torch.mean(torch.abs(reconstructs - faces))

            l1_loss = torch.mean(torch.abs(faces_fake - target_fulls))

            feature_loss = 0
            # feature_loss = torch.mean(torch.abs(FEN(faces_fake) - FEN(target_fulls)))

            lambda_rec = config.lambda_rec  # 2 to 4 to 8
            lambda_l1 = config.lambda_l1
            lambda_keypoint = config.lambda_keypoint   # 100 to 50
            lambda_fake = config.lambda_fake
            lambda_id = config.lambda_id
            lambda_feature = config.lambda_feature
            g_loss = lambda_fake*g_fake_loss \
                       + lambda_l1 * l1_loss + lambda_feature*feature_loss

            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            # Print out training information.
            if (i + 1) % 4 == 0:
                print("iter {} - d_real_loss {:.2}, d_fake_loss {:.2}, d_loss_gp {:.2}"
                      " g_fake_loss {:.2}, L1_loss {:.2}".format(i, real_loss.item(), fake_loss.item(), lambda_gp * d_loss_gp
                                                                                                     , lambda_fake*g_fake_loss.item(),
                                                                lambda_l1 * l1_loss.item()))

            sample_dir = "fusion-sample-{}".format(version)
            if not os.path.isdir(sample_dir):
                os.mkdir(sample_dir)
            if (i + 1) % 24 == 0:
                with torch.no_grad():
                    target_mask = target_masks[0]
                    fake_face = faces_fake[0]
                    target_full = target_fulls[0]
                    target_crop = target_crops[0]
                    origin_full = fullfaces[0]

                    sample_path_face = os.path.join(sample_dir, '{}-image-mask.jpg'.format(i + 1))
                    save_image(denorm(target_mask.data.cpu()), sample_path_face)

                    sample_path_fake = os.path.join(sample_dir, '{}-image-fake.jpg'.format(i + 1))
                    save_image(denorm(fake_face.data.cpu()), sample_path_fake)

                    sample_path_target = os.path.join(sample_dir, '{}-image-target-full.jpg'.format(i + 1))
                    save_image(denorm(target_full.data.cpu()), sample_path_target)

                    sample_path_predict_points = os.path.join(sample_dir, '{}-image-crop.jpg'.format(i + 1))
                    save_image(denorm(target_crop.data.cpu()), sample_path_predict_points)

                    sample_path_predict_full = os.path.join(sample_dir, '{}-image-origin_full.jpg'.format(i + 1))
                    save_image(denorm(origin_full.data.cpu()), sample_path_predict_full)

                    print('Saved real and fake images into {}...'.format(sample_path_face))


        # Save model checkpoints.
        model_save_dir = "fusion-ckpt-{}".format(version)

        if (i + 1) % 1000 == 0:
            if not os.path.isdir(model_save_dir):
                os.mkdir(model_save_dir)
            G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(i + 1))
            torch.save(G.state_dict(), G_path)
            D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(i + 1))
            torch.save(D.state_dict(), D_path)
            print('Saved model checkpoints into {}...'.format(model_save_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--resume_iter', type=int, default=14000)
    parser.add_argument('--version', type=str, default="256-level2")
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
