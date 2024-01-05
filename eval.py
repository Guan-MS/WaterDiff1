import argparse
import core.metrics as Metrics
from core.metrics import uciqe as UCIQE
from core.metrics import getUIQM as UIQM
from PIL import Image
import numpy as np
import glob
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default=r'D:\GMS\A\WaterDiff1\eval_results\82_UIEBB_100')
    parser.add_argument('-traget', '--path_traget', type=str,
                        default=r'D:\GMS\UI-data\UIEB_R90\test\ColorCorrection\target_256')
    parser.add_argument('-input', '--path_input', type=str,
                        default=r'D:\GMS\UI-data\UIEB_R90\test\ColorCorrection\input_256')

    args = parser.parse_args()

    #在同一文件夹下的数据
    target_names = list(glob.glob('{}/*_target.png'.format(args.path)))
    out_names = list(glob.glob('{}/*_output.png'.format(args.path)))
    # target_names = list(glob.glob('{}/*_GT.png'.format(args.path)))
    # out_names = list(glob.glob('{}/*_output.png'.format(args.path)))
    # target_names = list(glob.glob('{}/*_in.png'.format(args.path)))
    # out_names = list(glob.glob('{}/*_out3.png'.format(args.path)))
    #在不同一文件夹下的数据
    # target_names = list(glob.glob('{}/*.png'.format(args.path_traget)))
    # out_names = list(glob.glob('{}/*.png'.format(args.path_input)))


    target_names.sort()
    out_names.sort()
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_uiqm = 0.0
    avg_uciqe = 0.0
    idx = 0
    for target_name, out_name in zip(target_names, out_names):
        idx += 1
        ridx = target_name.rsplit("_target")[0]
        result = out_name
        fidx = out_name.rsplit("_out")[0]
        #assert ridx == fidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(ridx=ridx, fidx=fidx)

        target_img = np.array(Image.open(target_name))
        out_img = np.array(Image.open(out_name))
        psnr = Metrics.calculate_psnr(out_img, target_img)
        ssim = Metrics.calculate_ssim(out_img, target_img)
        uciqe = UCIQE(result)
        result = cv2.imread(result)
        uiqm = UIQM(result)

        avg_psnr += psnr
        avg_ssim += ssim
        avg_uciqe += uciqe
        avg_uiqm += uiqm
        if idx % 1 == 0:
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}, UIQM:{:.4f}, UCIQE:{:.4f}'.format(idx, psnr, ssim,uiqm,uciqe))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_uiqm = avg_uiqm / idx
    avg_uciqe = avg_uciqe / idx

    # log
    print('Avg_PSNR:{:.4f}, Avg_SSIM:{:.4f}, Avg_UIQM:{:.4f}, Avg_UCIQE:{:.4f}'.format(avg_psnr, avg_ssim, avg_uiqm, avg_uciqe))
    # print('Avg_PSNR: {:.4f}'.format(avg_psnr))
    # print('Avg_SSIM: {:.4f}'.format(avg_ssim))
    # print('Avg_UIQM: {:.4f}'.format(avg_uiqm))
    # print('Avg_UCIQE: {:.4f}'.format(avg_uciqe))
