import argparse
parser = argparse.ArgumentParser()

# Model configuration.
parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
parser.add_argument('--dim_neck', type=int, default=16)
parser.add_argument('--dim_emb', type=int, default=256)
parser.add_argument('--dim_pre', type=int, default=512)
parser.add_argument('--freq', type=int, default=16)

# Training configuration.
parser.add_argument('--data_dir', type=str, default='./spmel')
parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')

# Miscellaneous.
parser.add_argument('--log_step', type=int, default=10)

config = {}
args = parser.parse_args()
config = {key: parser.get_default(key) for key in vars(args)}
print(config)