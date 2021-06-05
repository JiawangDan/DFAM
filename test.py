import os
import torch
import argparse
import importlib
import torch.backends.cudnn as cudnn
from utils import *
from torch.autograd import Variable
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Pytorch Eval')
parser.add_argument('--model', type=str, default='VDSR_DFAM', help='The name of model.')
parser.add_argument('--scale', type=str, default=4, help='upscale factor')
parser.add_argument('--dataset', type=str, default='middlebury')
parser.add_argument('--upsample', action='store_true', default=False)
parser.add_argument('--rgb2y', action='store_true', default=False)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--testset_dir', type=str, default='../data/test', help='testset path')
parser.add_argument('--device', type=str, default='cuda')


def test(opt, test_loader, net, logger, out_path):
    net.eval()

    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        for iteration, (HR_left, HR_right, LR_left, LR_right) in enumerate(test_loader):
            img_name = test_loader.dataset.file_list[iteration]
            if opt.rgb2y:  # VDSR, SRCNN
                HR_left, HR_right, LR_left, LR_right, LR_left_y, LR_right_y = HR_left / 255, HR_right / 255, \
                    LR_left / 255, LR_right / 255, rgb2y(LR_left) / 255, rgb2y(LR_right) / 255
                HR_left, HR_right, LR_left, LR_right, LR_left_y, LR_right_y = \
                    Variable(HR_left).to(opt.device), Variable(HR_right).to(opt.device), \
                    Variable(LR_left).to(opt.device), Variable(LR_right).to(opt.device), \
                    Variable(LR_left_y).to(opt.device), Variable(LR_right_y).to(opt.device)
                SR_left_y, SR_right_y, _, _ = net(LR_left_y, LR_right_y)
                SR_left = img_transfer(LR_left, SR_left_y)
                SR_right = img_transfer(LR_right, SR_right_y)
                SR_left = torch.clamp(SR_left, 0, 1)
                SR_right = torch.clamp(SR_right, 0, 1)
            else:  # SRResNet
                HR_left, HR_right, LR_left, LR_right = HR_left / 255, HR_right / 255, LR_left / 255, LR_right / 255
                HR_left, HR_right, LR_left, LR_right = \
                    Variable(HR_left).to(opt.device), Variable(HR_right).to(opt.device), \
                    Variable(LR_left).to(opt.device), Variable(LR_right).to(opt.device)
                SR_left, SR_right, _, _ = net(LR_left, LR_right)
                SR_left = torch.clamp(SR_left, 0, 1)
                SR_right = torch.clamp(SR_right, 0, 1)

            psnr = cal_psnr(HR_left[:, :, :, 64:].data.cpu().numpy(), SR_left[:, :, :, 64:].data.cpu().numpy())
            ssim = cal_ssim(HR_left[0, :, :, 64:].data.cpu().numpy(), SR_left[0, :, :, 64:].data.cpu().numpy())
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            logger.info('{}, psnr: {}, ssim: {}'.format(img_name, psnr, ssim))

            if out_path:
                save_img(torch.squeeze(SR_left.data.cpu(), 0), os.path.join(out_path, '{}_sr_left.png'.format(img_name)))
                save_img(torch.squeeze(SR_right.data.cpu(), 0), os.path.join(out_path, '{}_sr_right.png'.format(img_name)))
                save_img(torch.squeeze(HR_left.data.cpu(), 0), os.path.join(out_path, '{}_hr_left.png'.format(img_name)))
                save_img(torch.squeeze(HR_right.data.cpu(), 0), os.path.join(out_path, '{}_hr_right.png'.format(img_name)))
        logger.info('===> Avg. PSNR: {:.5f} dB, Avg. SSIM: {:.5f}'.format(np.mean(psnr_list), np.mean(ssim_list)))


def test_SISR(opt, test_loader, net, logger, out_path):
    net.eval()

    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        for iteration, (HR_left, HR_right, LR_left, LR_right) in enumerate(test_loader):
            img_name = test_loader.dataset.file_list[iteration]
            if opt.rgb2y:  # VDSR, SRCNN
                HR_left, HR_right, LR_left, LR_right, LR_left_y, LR_right_y = HR_left / 255, HR_right / 255, \
                    LR_left / 255, LR_right / 255, rgb2y(LR_left) / 255, rgb2y(LR_right) / 255
                HR_left, HR_right, LR_left, LR_right, LR_left_y, LR_right_y = \
                    Variable(HR_left).to(opt.device), Variable(HR_right).to(opt.device), \
                    Variable(LR_left).to(opt.device), Variable(LR_right).to(opt.device), \
                    Variable(LR_left_y).to(opt.device), Variable(LR_right_y).to(opt.device)
                SR_left_y = net(LR_left_y)
                SR_left = img_transfer(LR_left, SR_left_y)
                SR_left = torch.clamp(SR_left, 0, 1)
            else:  # SRResNet, RCAN
                HR_left, HR_right, LR_left, LR_right = HR_left / 255, HR_right / 255, LR_left / 255, LR_right / 255
                HR_left, HR_right, LR_left, LR_right = \
                    Variable(HR_left).to(opt.device), Variable(HR_right).to(opt.device), \
                    Variable(LR_left).to(opt.device), Variable(LR_right).to(opt.device)
                SR_left = net(LR_left)
                SR_left = torch.clamp(SR_left, 0, 1)

            psnr = cal_psnr(HR_left[:, :, :, 64:].data.cpu().numpy(), SR_left[:, :, :, 64:].data.cpu().numpy())
            ssim = cal_ssim(HR_left[0, :, :, 64:].data.cpu().numpy(), SR_left[0, :, :, 64:].data.cpu().numpy())
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            logger.info('{}, psnr: {}, ssim: {}'.format(img_name, psnr, ssim))

            if out_path:
                save_img(torch.squeeze(SR_left.data.cpu(), 0), os.path.join(out_path, '{}_sr_left.png'.format(img_name)))
                save_img(torch.squeeze(HR_left.data.cpu(), 0), os.path.join(out_path, '{}_hr_left.png'.format(img_name)))
        logger.info('===> Avg. PSNR: {:.5f} dB, Avg. SSIM: {:.5f}'.format(np.mean(psnr_list), np.mean(ssim_list)))


def main():
    opt = parser.parse_args()

    out_path = os.path.join('result', opt.model + '_x' + str(opt.scale), opt.dataset)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    logger = get_logger(os.path.join(out_path, opt.model + '_x' + str(opt.scale) + '_result.txt'))
    logger.info(opt)

    cudnn.benchmark = True

    opt.dataset_dir = os.path.join(opt.testset_dir, opt.dataset)
    test_set = TestSetLoader(opt, logger)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = get_model(opt)

    if opt.checkpoint == '':
        raise Exception('The checkpoint is null, please set the checkpoint')
    state = torch.load(opt.checkpoint)
    net.load_state_dict(state['model'])
    net.eval()

    if 'DFAM' in opt.model:
        test(opt, test_loader, net, logger, out_path)
    else:
        test_SISR(opt, test_loader, net, logger, out_path)


if __name__ == '__main__':
    main()
