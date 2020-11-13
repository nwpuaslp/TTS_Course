import torch
from utils.dataset import WaveRNNDataset, WaveRNNCollate
from torch.utils.data import DataLoader
import torch.nn.parallel.data_parallel as parallel
import torch.optim as optim
import torch.nn as nn
import argparse
import os
from models.model import Model
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Optimizer
from utils.util import ExponentialMovingAverage, apply_moving_average, register_model_to_ema


def create_model(args):
    model = Model(quantization_channels=args.quantization_channels,
                  gru_channels=896,
                  fc_channels=896,
                  lc_channels=args.local_condition_dim,
                  upsample_factor=(5, 5, 8),
                  use_gru_in_upsample=True)

    return model


def save_checkpoint(args, model, optimizer, step, ema=None):
    checkpoint_path = os.path.join(args.checkpoint_dir,
                                   "model.ckpt-{}.pt".format(step))

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": step
        }, checkpoint_path)

    if ema is not None:
        ema_checkpoint_path = os.path.join(args.ema_checkpoint_dir,
                                           "model.ckpt-{}.ema".format(step))
        averge_model = clone_as_averaged_model(args, model, ema)
        torch.save(
            {
                "model": averge_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": step
            }, ema_checkpoint_path)

    print("Saved checkpoint: {}".format(checkpoint_path))

    with open(os.path.join(args.checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}.pt".format(step))

    with open(os.path.join(args.ema_checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}.ema".format(step))


def attempt_to_restore(model, optimizer, checkpoint_dir, use_cuda):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
    else:
        global_step = 0

    return global_step


def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)

    return checkpoint


def clone_as_averaged_model(args, model, ema):
    device = torch.device("cuda" if args.use_cuda else "cpu")

    assert ema is not None
    averaged_model = create_model(args).to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()

    return averaged_model


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.ema_checkpoint_dir, exist_ok=True)

    train_dataset = WaveRNNDataset(args.data_dir)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    model = create_model(args)

    print(model)

    num_gpu = torch.cuda.device_count() if args.use_cuda else 1

    model.train(mode=True)

    global_step = 0

    parameters = list(model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)

    writer = SummaryWriter(args.checkpoint_dir)

    model.to(device)

    if args.resume is not None:
        restore_step = attempt_to_restore(model, optimizer, args.resume,
                                          args.use_cuda)
        global_step = restore_step

    ema = ExponentialMovingAverage(args.ema_decay)
    register_model_to_ema(model, ema)

    customer_optimizer = Optimizer(optimizer, args.learning_rate, global_step,
                                   args.warmup_steps, args.decay_learning_rate)

    criterion = nn.NLLLoss().to(device)

    for epoch in range(args.epochs):

        collate = WaveRNNCollate(upsample_factor=args.hop_length,
                                 condition_window=args.condition_window)

        train_data_loader = DataLoader(train_dataset,
                                       collate_fn=collate,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True,
                                       pin_memory=True)

        #train one epoch
        for batch, (coarse, fine, condition) in enumerate(train_data_loader):

            batch_size = int(condition.shape[0] // num_gpu * num_gpu)

            coarse = coarse[:batch_size, :].to(device)
            fine = fine[:batch_size, :].to(device)
            condition = condition[:batch_size, :, :].to(device)
            inputs = torch.cat([
                coarse[:, :-1].unsqueeze(-1), fine[:, :-1].unsqueeze(-1),
                coarse[:, 1:].unsqueeze(-1)
            ],
                               dim=-1)
            inputs = 2 * inputs.float() / 255 - 1.0

            if num_gpu > 1:
                out_c, out_f, _ = parallel(model, (inputs, condition))
            else:
                out_c, out_f, _ = model(inputs, condition)

            loss_c = criterion(out_c.transpose(1, 2).float(), coarse[:, 1:])
            loss_f = criterion(out_f.transpose(1, 2).float(), fine[:, 1:])
            loss = loss_c + loss_f

            global_step += 1
            customer_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
            customer_optimizer.step_and_update_lr()
            model.after_update()

            if ema is not None:
                apply_moving_average(model, ema)

            if global_step % args.log_step == 0:
                print("Step: {} --loss_c: {:.3f} --loss_f: {:.3f} --Lr: {:g}".
                      format(global_step, loss_c, loss_f,
                             customer_optimizer.lr))

            if global_step % args.checkpoint_step == 0:
                save_checkpoint(args, model, optimizer, global_step, ema)

            if global_step % args.summary_step == 0:
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("loss_c", loss_c.item(), global_step)
                writer.add_scalar("loss_f", loss_f.item(), global_step)


def main():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='data',
                        help='Directory of training data')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of dataloader workers.')
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default="logdir",
                        help="Directory to save model")
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help="The model name to restore")
    parser.add_argument('--checkpoint_step', type=int, default=1000)
    parser.add_argument('--summary_step', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--warmup_steps', type=int, default=50000)
    parser.add_argument('--decay_learning_rate', type=float, default=0.5)
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--condition_window', type=int, default=20)
    parser.add_argument('--quantization_channels', type=int, default=256)
    parser.add_argument('--hop_length', type=int, default=200)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--ema_checkpoint_dir', type=str, default="ema_logdir")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
