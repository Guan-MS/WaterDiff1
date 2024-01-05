import argparse
import os
from ldm.data import testsets_vq as testsets_vqm
from datasets import testsets

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--exp', type=str, default="configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml")
parser.add_argument('--dataset', type=str, default='Middlebury')
parser.add_argument('--metrics', nargs='+', type=str, default=['FloLPIPS'])
parser.add_argument('--data_dir', type=str, default=r'D:\GMS\UI-data\UIEB_R90\test\10')
parser.add_argument('--out_dir', type=str, default='eval_results')
parser.add_argument('--resume', dest='resume', default=False, action='store_true')

# .add_argument('--dataset', type=str, default="UIEB", help="Underwater datasets")
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
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # initialise test set
    print('Testing on dataset: ', args.dataset)
    test_dir = os.path.join(args.out_dir, args.dataset)

    test_db = getattr(testsets_vqm, args.test_dataloder)(os.path.join(args.data_dir), args.image_size, args.data_len, args.phase, args.batch_size,
                                              args.shuffle, args.num_workers)

    test_db.eval(metrics=args.metrics, output_dir=test_dir, resume=args.resume)


if __name__ == '__main__':
    main()