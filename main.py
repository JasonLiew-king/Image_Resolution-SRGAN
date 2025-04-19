#from mode import *
from test_mode import *
import argparse
from torch.cuda.amp import autocast


parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

# --- Arguments ---
parser.add_argument("--LR_path", type=str, default='C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/train/train_lr')
parser.add_argument("--GT_path", type=str, default='C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/train/train_hr')
parser.add_argument("--val_LR_path", type=str, default='C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/val/val_lr')
parser.add_argument("--val_GT_path", type=str, default='C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/val/val_hr')
parser.add_argument("--res_num", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--L2_coeff", type=float, default=1.0)
parser.add_argument("--adv_coeff", type=float, default=1e-3)
parser.add_argument("--tv_loss_coeff", type=float, default=0.0)
parser.add_argument("--pre_train_epoch", type=int, default=15)
parser.add_argument("--fine_train_epoch", type=int, default=35)
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--patch_size", type=int, default=24)
parser.add_argument("--feat_layer", type=str, default='relu5_4')
parser.add_argument("--vgg_rescale_coeff", type=float, default=0.006)
#parser.add_argument("--fine_tuning", type=str2bool, default=False)
parser.add_argument('--fine_tuning', action='store_true', help='Enable fine-tuning from pretrained generator')
parser.add_argument("--in_memory", type=str2bool, default=False)
parser.add_argument("--generator_path", type=str)
parser.add_argument("--mode", type=str, default='pretrain')  # now accepts: pretrain, train_gan, test, test_only
parser.add_argument("--tile_size", type=int, default=96, help="Tile size for LR input during test_only")
parser.add_argument("--overlap", type=int, default=16, help="Overlap between tiles during test_only")
parser.add_argument("--output_path", type=str, default="C:/Users/User/Desktop/SRGAN - Copy/div2k_train_val/outputs/", help="Directory to save output images")


args = parser.parse_args()

# --- Mode selection ---
if args.mode == 'pretrain':
    pretrain_generator(args)
    
elif args.mode == 'train':
    train(args)
    
elif args.mode == 'test':
    test(args)

elif args.mode == 'train_gan':
    train_gan(args)

elif args.mode == 'test_only':
    test_only(args)

else:
    raise ValueError("Invalid mode. Choose from: pretrain, train_gan, test, test_only.")
