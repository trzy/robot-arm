#
# __main__.py
# Bart Trzynadlowski
#
# ACT model training code. Adapted directly from the original ACT codebase: 
# https://github.com/tonyzhaozh/act/
#
# Ingests datasets produced by the server module (see server/__main__.py), which are stored in
# dataset directories as follows:
#
#   dataset_dir/example-0/
#       data.hdf5
#   dataset_dir/example-1/
#       data.hdf5
#   ...
#
# Usage:
#
#   - To train using the default hyperparameters using a dataset named "cube" with checkpoints
#     output to "cube/checkpoints":
#
#       python -m act --dataset-dir=cube --checkpoint-dir=cube/checkpoints
#
#   - Increasing the batch size to 64 and learning rate to 5e-5:
#
#       python -m act --dataset-dir=cube --checkpoint-dir=cube/checkpoints --batch-size-64 --lr=5e-5
#

import argparse
from argparse import Namespace
from copy import deepcopy
import os
import pickle
from typing import Any, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_data
from .utils import compute_dict_mean, set_seed, detach_dict
from .policy import ACTPolicy, CNNMLPPolicy

def main(options: Namespace):
    set_seed(1)

    ckpt_dir = options.checkpoint_dir
    policy_class = options.policy_class

    camera_names = [ "top" ]    # single camera (for now)
    state_dim = 5               # robot has 5 motors
    lr_backbone = 1e-5
    backbone = 'resnet18'

    policy_config = deepcopy(options)
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        setattr(policy_config, 'num_queries', options.chunk_size)
        setattr(policy_config, 'lr_backbone', lr_backbone)
        setattr(policy_config, 'backbone', backbone)
        setattr(policy_config, 'enc_layers', enc_layers)
        setattr(policy_config, 'dec_layers', dec_layers)
        setattr(policy_config, 'nheads', nheads)
        setattr(policy_config, 'camera_names', camera_names)
    elif policy_class == 'CNNMLP':
        setattr(policy_config, 'lr_backbone', lr_backbone)
        setattr(policy_config, 'backbone', backbone)
        setattr(policy_config, 'num_queries', 1)
        setattr(policy_config, 'camera_names', camera_names)
    else:
        raise NotImplementedError

    config = {
        'num_epochs': options.num_epochs,
        'ckpt_dir': ckpt_dir,
        'state_dim': state_dim,
        'lr': options.lr,
        'policy_class': policy_class,
        'policy_config': policy_config,
        'seed': options.seed,
        'camera_names': camera_names,
    }

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir=options.dataset_dir,
        chunk_size=options.chunk_size,
        camera_names=camera_names,
        batch_size_train=options.batch_size,
        batch_size_val=options.batch_size
    )

    # Save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class: str, policy_config: Namespace):
    if policy_class.upper() == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class.upper() == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class: str, policy):
    if policy_class.upper() == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class.upper() == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # Validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # Training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # Backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 1000 == 0 and epoch > 3000:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # Save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # Save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("act")

    # File source and destination
    parser.add_argument('--checkpoint-dir', action='store', type=str, help='Directory to write checkpoints to', required=True)
    parser.add_argument('--dataset-dir', action='store', type=str, help='Directory from which to read episodes (i.e., dataset_dir/example-*/data.hdf5)', required=True)

    # Training parameters
    parser.add_argument('--policy-class', action='store', type=str, default="ACT", help='Policy class: ACT or CNNMLP')
    parser.add_argument('--batch-size', action='store', type=int, default=8, help='Batch size')
    parser.add_argument('--num-epochs', action='store', type=int, default=8000, help='Number of epochs to train for')
    parser.add_argument('--lr', action='store', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--seed', action='store', type=int, default=42, help='Random seed')

    # ACT
    parser.add_argument('--kl-weight', action='store', type=int, default=10, help='ACT model KL weight')
    parser.add_argument('--chunk-size', action='store', type=int, default=100, help='ACT model chunk size')
    parser.add_argument('--hidden-dim', action='store', type=int, default=512, help='ACT model hidden dimension')
    parser.add_argument('--dim-feedforward', action='store', type=int, default=32000, help='ACT model feed-forward dimension')

    options = parser.parse_args()
    main(options)
