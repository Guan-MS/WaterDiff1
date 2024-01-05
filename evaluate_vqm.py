import argparse
import os
from ldm.data import testsets_vq as testsets_vqm
from datasets import testsets

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--exp', type=str, default="logs/2023-08-04T13-44-53_c/checkpoints/epoch=000050.ckpt")
parser.add_argument('--dataset', type=str, default='Middlebury')
parser.add_argument('--metrics', nargs='+', type=str, default=['FloLPIPS'])
parser.add_argument('--data_dir', type=str, default=r'D:\GMS\UI-data\UIEB_R90\test\10')
parser.add_argument('--out_dir', type=str, default='eval_results')
parser.add_argument('--resume', dest='resume', default=False, action='store_true')

#.add_argument('--dataset', type=str, default="UIEB", help="Underwater datasets")


# sampler args
parser.add_argument('--use_ddim', dest='use_ddim', default="use_ddim", action='store_true')
parser.add_argument('--ddim_eta', type=float, default=1.0)
parser.add_argument('--ddim_steps', type=int, default=100)
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
    model = args.exp
    print('Evaluating model:', model)

    # setup output dirs
    assert os.path.exists(args.out_dir), 'Frames not previously interpolated!'
    
    # initialise test set
    print('Testing on dataset: ', args.dataset)
    test_dir = os.path.join(args.out_dir, args.dataset)
    assert os.path.exists(test_dir), f'{args.dataset} not pre-computed!'

    if args.dataset.split('_')[0] in ['VFITex', 'Ucf101', 'Davis90']:
        db_folder = args.dataset.split('_')[0].lower()
    else:
        db_folder = args.dataset.lower()

    test_db = getattr(testsets_vqm, args.dataset)(os.path.join(args.data_dir, db_folder))
    test_db.eval(metrics=args.metrics, output_dir=test_dir, resume=args.resume)



if __name__ == '__main__':
    main()