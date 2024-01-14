import argparse
import datetime
import time
import torch.backends.cudnn as cudnn
from modeling_pretrain import pretrain_mae_small_patch16_224, pretrain_mae_base_patch16_224, pretrain_mae_large_patch16_224
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer
from engine_for_pretraining import train_one_epoch, test_one_epoch
import mae_utils
from data_loaders import *
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--time_win', type=int, default=1, help='time window')
    parser.add_argument('--down_sample', default=4, type=int, help='down sample rate')
    parser.add_argument('--sample_freq', default=1000, type=int, help='bci sample frequency')
    parser.add_argument('--device', default='cuda', type=str, help='training device')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.9, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=(62, 250), type=int,
                        help='images input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=False, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=(0.9, 0.95), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=1000, metavar='N',#1200
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Dataset parameters
    parser.add_argument('--seed', default=2023, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


def main(args):
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    model = pretrain_mae_small_patch16_224(pretrained=False,
        drop_path_rate=args.drop_path, attn_drop_rate=0., drop_rate=0.)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size[0] // patch_size[0], args.input_size[1] // patch_size[1])
    args.patch_size = patch_size

    data_loader_train, data_loader_valid = data_generator_np(args.data_path, args.batch_size, args.window_size, args.mask_ratio, win_train=args.time_win, down_sample=args.down_sample,
                          sample_freq=args.sample_freq, device=args.device)

    num_training_steps_per_epoch = len(data_loader_train)


    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = mae_utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))


    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % args.batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (args.batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model_without_ddp)
    print("Use step level LR & WD scheduler!")

    lr_schedule_values = mae_utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps, )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_test_loss = 1e+10
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            patch_size=patch_size,
            normlize_target=args.normlize_target,
        )
        test_loss = test_one_epoch(
            model, data_loader_valid,
            device, epoch, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            patch_size=patch_size,
            normlize_target=args.normlize_target,
        )
        if args.output_dir and test_loss <= best_test_loss:
            best_test_loss = test_loss
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                mae_utils.save_model(args=args, model=model, epoch=epoch)

        if args.output_dir and mae_utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    model_name = 'frft_mae'
    task_list = ['ssvep', 'mi']
    for task in task_list:
        session_list = ['session1', 'session2']
        for session in session_list:
            sub_list = ['s6', 's14', 's20', 's28', 's39']
            print(len(sub_list))
            for subj_index in range(len(sub_list)):
                subj_number = sub_list[subj_index]
                print(subj_number)
                opts.data_path = os.path.join('./processed_{}_data/'.format(task), session, subj_number)
                opts.output_dir = os.path.join('./model_ratio_0.9_{}'.format(model_name), task, session, subj_number)
                opts.log_dir = None
                if opts.output_dir:
                    Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
                main(opts)

