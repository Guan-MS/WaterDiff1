import argparse
import os
import torch
from functools import partial
from omegaconf import OmegaConf
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
#from ldm.data import testsets
from datasets import testsets

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--config', type=str, default="configs/ldm/ldm.yaml")
parser.add_argument('--ckpt', type=str, default="logs/ldm_bao/checkpoints/epoch=000068.ckpt")
parser.add_argument('--dataset', type=str, default="UIEB_200", help="Underwater datasets")
parser.add_argument('--data_dir', type=str, default=r'D:\GMS\UI-data\UIEB_R90\test')
parser.add_argument('--out_dir', type=str, default='eval_results')

# sampler args
parser.add_argument('--use_ddim', dest='use_ddim', default="use_ddim", action='store_true')
parser.add_argument('--ddim_eta', type=float, default=1.0)
parser.add_argument('--ddim_steps', type=int, default=200)
# dataset
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--data_len', type=int, default=-1)
parser.add_argument('--phase', type=str, default="test")
parser.add_argument('--shuffle', type=str, default="False")
parser.add_argument('--test_dataloder', type=str, default="dataset")

def main():

    args = parser.parse_args()
    
    # initialise model
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(args.ckpt)['state_dict'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = model.eval()
    print('Model loaded successfully')

    # set up sampler
    if args.use_ddim:
        ddim = DDIMSampler(model)
        sample_func = partial(ddim.sample, S=args.ddim_steps, eta=args.ddim_eta, verbose=False)
    else:
        sample_func = partial(model.sample_ddpm, return_intermediates=False, verbose=False)

    # setup output dirs
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # initialise test set
    print('Testing on dataset: ', args.dataset)
    test_dir = os.path.join(args.out_dir, args.dataset)

    test_db = getattr(testsets, args.test_dataloder)(os.path.join(args.data_dir), args.image_size, args.data_len, args.phase, args.batch_size,
                                              args.shuffle, args.num_workers)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    test_db.eval(model, sample_func, output_dir=test_dir, device=device)



if __name__ == '__main__':
    main()