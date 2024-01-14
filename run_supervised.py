import argparse
import datetime
import time
import torch.backends.cudnn as cudnn
import json
from data_loaders_finetune import *
from pathlib import Path
from collections import OrderedDict
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from modeling_finetune import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224
from engine_for_finetuning import train_one_epoch, evaluate
import mae_utils


def get_args():
    parser = argparse.ArgumentParser('MAE fine-tuning and evaluation script for image classification', add_help=False)
    parser.add_argument('--eval', default=False, type=bool)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=5000, type=int)#5000
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
##########
    parser.add_argument('--time_win', type=int, default=1, help='time window')
    parser.add_argument('--down_sample', default=4, type=int, help='down sample rate')
    parser.add_argument('--sample_freq', default=1000, type=int, help='bci sample frequency')
#########
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=(62, 250), type=int,
                        help='images input size')
    parser.add_argument('--drop', type=float, default=0.5, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.3, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.3, metavar='PCT',
                        help='Drop path rate (default: 0.)')
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
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
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)#0.75
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--save_ckpt', default=True, type=bool)
    parser.set_defaults(save_ckpt=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = mae_utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    model = vit_small_patch16_224(pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,  # proj_drop and pos_drop
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size[0] // 1, args.input_size[1] // 250)
    args.patch_size = patch_size
    data_loader_train, data_loader_val = data_generator_np(args.data_path, args.batch_size,
                                                             win_train=args.time_win,
                                                             down_sample=args.down_sample,
                                                             sample_freq=args.sample_freq, device=args.device)
    if args.finetune:
        checkpoint_model = torch.load(args.finetune, map_location='cpu')
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        mae_utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params:', n_parameters)
    num_training_steps_per_epoch = len(data_loader_train)
    num_layers = model.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))#14-(0..13)
    else:
        assigner = None
    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))
    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)
    optimizer = create_optimizer(
            args, model, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
    print("Use step level LR scheduler!")
    lr_schedule_values = mae_utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,)
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = mae_utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        exit(0)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,)
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device)
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    mae_utils.save_model(
                        args=args, epoch=epoch, model=model)
            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        if args.output_dir and mae_utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if max_accuracy==100. and test_stats["acc1"] > 90:
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return max_accuracy


if __name__ == '__main__':
    opts = get_args()
    target_task = 'mi'
    session = 'session2'
    if target_task == 'mi':
        opts.nb_classes = 2
    elif target_task == 'ssvep':
        opts.nb_classes = 4
    sub_list = os.listdir(os.path.join('./processed_{}_data'.format(target_task), session))
    sub_list = sorted(sub_list)
    print(len(sub_list))
    record_file = open('./record_supervised_{}_{}.txt'.format(target_task, session), 'w')
    for subj_index in range(len(sub_list)):
        subj_number = sub_list[subj_index]
        print(subj_number)
        opts.data_path = os.path.join('./processed_{}_data/'.format(target_task), session, subj_number)
        opts.finetune = False
        opts.output_dir = './finetune_model'
        opts.log_dir = None
        if opts.output_dir:
            Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
        acc = main(opts)
        record_file.write(subj_number+' '+str(acc)+'\n')
    record_file.close()
