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
#       python -m act --train --dataset-dir=cube --checkpoint-dir=cube/checkpoints
#
#   - Increasing the batch size to 64 and learning rate to 5e-5:
#
#       python -m act --train --dataset-dir=cube --checkpoint-dir=cube/checkpoints --batch-size=64
#       --lr=5e-5
#
#   - Performing inference on a checkpoint:
#
#       python -m act --infer --checkpoint-file=cube/checkpoints/policy_best.ckpt
#

import argparse
from argparse import Namespace
import asyncio
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import os
import pickle
import timeit
from typing import DefaultDict, List

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_data
from .utils import compute_dict_mean, set_seed, detach_dict
from .policy import ACTPolicy, CNNMLPPolicy


####################################################################################################
# Constants
####################################################################################################

NUM_MOTORS = 5              # robot has 5 motors


####################################################################################################
# Training and Inference
####################################################################################################

@dataclass
class Observation:
    qpos: np.ndarray
    image: np.ndarray

def train(options: Namespace):
    train_dataloader, val_dataloader, stats, _, camera_names = load_data(
        dataset_dir=options.dataset_dir,
        chunk_size=options.chunk_size,
        batch_size_train=options.batch_size,
        batch_size_val=options.batch_size
    )

    config = {
        "num_epochs": options.num_epochs,
        "ckpt_dir": options.checkpoint_dir,
        "state_dim": NUM_MOTORS,
        "lr": options.lr,
        "policy_class": options.policy_class,
        "policy_config": make_policy_config(options=options, camera_names=camera_names),
        "seed": options.seed,
        "camera_names": camera_names,
    }

    # Save dataset stats
    if not os.path.isdir(options.checkpoint_dir):
        os.makedirs(options.checkpoint_dir)
    stats_path = os.path.join(options.checkpoint_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(options.checkpoint_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")

async def infer(options: Namespace, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
    try:
        await _infer(options=options, input_queue=input_queue, output_queue=output_queue)
    except Exception as e:
        print(f"Error: {e}")

def get_camera_names(num_cameras: int):
    return [ f"cam{i}" for i in range(num_cameras) ]

async def _infer(options: Namespace, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
    policy_class = options.policy_class.upper()
    policy_config = make_policy_config(options=options, camera_names=get_camera_names(num_cameras=options.num_cameras))
    policy = make_policy(policy_class=options.policy_class, policy_config=policy_config)

    # Load checkpoint
    loading_status = policy.load_state_dict(torch.load(options.checkpoint_file))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {options.checkpoint_file}')

    # Load dataset_stats.pkl and create functions to process data going into and actions coming out
    # of the policy
    checkpoint_dir = os.path.dirname(options.checkpoint_file)
    stats_file = os.path.join(checkpoint_dir, "dataset_stats.pkl")
    with open(stats_file, mode="rb") as fp:
        stats = pickle.load(fp)
    pre_process_fn = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process_fn = lambda a: a * stats["action_std"] + stats["action_mean"]

    # How often to infer
    query_frequency = policy_config.num_queries
    if options.temporal_aggregation:
        query_frequency = 1
    num_queries = policy_config.num_queries

    # Inference loop
    all_time_actions = np.zeros((num_queries, num_queries, NUM_MOTORS))
    with torch.inference_mode():
        t = 0
        while True:
            # Get observation
            observation = await input_queue.get()
            if not isinstance(observation, Observation):
                # Reset state
                t = 0
                all_time_actions = np.zeros(all_time_actions.shape)
                continue
            if observation.image.shape[0] != options.num_cameras:
                print(f"Error: Received observation with {observation.image.shape[0]} camera images but model expects {options.num_cameras}")
                continue
            qpos_numpy = observation.qpos
            curr_image = prepare_image(frame=observation.image)
            qpos = pre_process_fn(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            # Query the policy
            if policy_class == "ACT":
                if t % query_frequency == 0:
                    print(f"t={t} infer (query_frequency={query_frequency}) image={observation.image.shape}")
                    all_actions = policy(qpos, curr_image)
                else:
                    print(f"t={t} sample {t%query_frequency} image={observation.image.shape}")

                if options.temporal_aggregation:
                    # Temporal aggregation code fixed by: @Mankaran32
                    all_actions = all_actions.squeeze(0).cpu().numpy()
                    all_time_actions[0:num_queries-1] = all_time_actions[1:]
                    all_time_actions[-1, :] = all_actions

                    # Generate diagonal indices with offset
                    diagonal_indices = np.arange(num_queries)

                    # Add the offset to the diagonal indices
                    diagonal_indices_with_offset = np.arange(num_queries)[::-1]

                    # Create a (50, NUM_MOTORS) array by extracting the diagonal elements with offset
                    action_count = max(1, min(t, num_queries))
                    actions_for_curr_step = all_time_actions[diagonal_indices_with_offset, diagonal_indices][:action_count]

                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(action_count))[::-1]
                    exp_weights = exp_weights / exp_weights.sum()

                    raw_action = (actions_for_curr_step * exp_weights.reshape(-1, 1)).sum(axis=0)
                else:
                    raw_action = all_actions[:, t % query_frequency]
                    raw_action = raw_action.squeeze(0).cpu().numpy()
            elif policy_class == "CNNMLP":
                raw_action = policy(qpos, curr_image)
                raw_action = raw_action.squeeze(0).cpu().numpy()
            else:
                raise NotImplementedError

            # Post-process actions
            action = post_process_fn(raw_action)
            target_qpos = action

            # Send
            await output_queue.put(target_qpos)

            # Next
            t += 1

def prepare_image(frame: np.ndarray) -> torch.Tensor:
    frame = frame.transpose([0, 3, 1, 2])  # (num_cameras,height,width,channels) -> (num_cameras,channels,height,width)
    return torch.from_numpy(frame / 255.0).float().cuda().unsqueeze(0)

def make_policy_config(options: Namespace, camera_names: List[str]) -> Namespace:
    policy_config = deepcopy(options)
    lr_backbone = 1e-5
    backbone = "resnet18"
    if options.policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        setattr(policy_config, "num_queries", options.chunk_size)
        setattr(policy_config, "lr_backbone", lr_backbone)
        setattr(policy_config, "backbone", backbone)
        setattr(policy_config, "enc_layers", enc_layers)
        setattr(policy_config, "dec_layers", dec_layers)
        setattr(policy_config, "nheads", nheads)
        setattr(policy_config, "camera_names", camera_names)
    elif options.policy_class == "CNNMLP":
        setattr(policy_config, "lr_backbone", lr_backbone)
        setattr(policy_config, "backbone", backbone)
        setattr(policy_config, "num_queries", 1)
        setattr(policy_config, "camera_names", camera_names)
    else:
        raise NotImplementedError
    return policy_config

def make_policy(policy_class: str, policy_config: Namespace):
    if policy_class.upper() == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class.upper() == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class: str, policy):
    if policy_class.upper() == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class.upper() == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data[0:4]
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")
        # Validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # Training
        timings: DefaultDict[str, List[float]] = defaultdict(list)
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            t0 = timeit.default_timer()
            forward_dict = forward_pass(data, policy)
    
            # Backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            
            t1 = timeit.default_timer()
            t_data_loader = torch.sum(data[-1]).item()  # sum entire batch of timing results
            timings["data"].append(t_data_loader)
            timings["model"].append(t1 - t0)

        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        if epoch % 1000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

        print_timings(timings=timings)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

    # Save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info

def print_timings(timings: DefaultDict[str, List[float]]):
    longest_measurement_name_length = max([ len(measurement_name) for measurement_name in timings.keys()])
    total_time = np.sum([ np.sum(timings[measurement_name]) for measurement_name in timings ])
    print("")
    print("Timings")
    print("-------")
    for measurement_name, measurements in timings.items():
        seconds = np.sum(measurements)
        millis = seconds / 1e-3
        pct_of_total = 100.0 * (seconds / total_time)
        padding = " " * (longest_measurement_name_length - len(measurement_name))
        print(f"{measurement_name}{padding} = {millis:.1f} ms ({pct_of_total:.1f}%)")
    print("")

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # Save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label="train")
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label="validation")
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


####################################################################################################
# Server
####################################################################################################

from annotated_types import Len
import base64
from typing import Annotated, List

from pydantic import BaseModel

from server.networking import handler, MessageHandler, Session, TCPServer

# HelloMessage is also used to reset inference process
class HelloMessage(BaseModel):
    message: str

class InferenceRequestMessage(BaseModel):
    motor_radians: Annotated[List[float], Len(min_length=5, max_length=5)]
    frames: List[str]

class InferenceResponseMessage(BaseModel):
    target_motor_radians: Annotated[List[float], Len(min_length=5, max_length=5)]

class InferenceServer(MessageHandler):
    def __init__(self, port: int, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        super().__init__()
        self.sessions = set()
        self._server = TCPServer(port=port, message_handler=self)
        self._input_queue = input_queue
        self._output_queue = output_queue

    async def run(self):
        await asyncio.gather(self._server.run(), self._send_results())

    async def _send_results(self):
        while True:
            target_motor_radians = await self._output_queue.get()
            msg = InferenceResponseMessage(target_motor_radians=target_motor_radians)
            for session in self.sessions:
                await session.send(msg)

    async def on_connect(self, session: Session):
        print("Connection from: %s" % session.remote_endpoint)
        await session.send(HelloMessage(message = "Hello from ACT inference server"))
        self.sessions.add(session)

    async def on_disconnect(self, session: Session):
        print("Disconnected from: %s" % session.remote_endpoint)
        self.sessions.remove(session)

    @handler(HelloMessage)
    async def handle_HelloMessage(self, session: Session, msg: HelloMessage, timestamp: float):
        print("Hello received: %s" % msg.message)
        await self._input_queue.put(msg)

    @handler(InferenceRequestMessage)
    async def handle_InferenceRequestMessage(self, session: Session, msg: InferenceRequestMessage, timestamp: float):
        frames = []
        for i in range(len(msg.frames)):
            jpeg = np.frombuffer(buffer=base64.b64decode(msg.frames[i]), dtype=np.uint8)
            frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            frames.append(frame)
        frame = np.stack(frames, axis=0)
        motor_radians = np.array(msg.motor_radians)
        await self._input_queue.put(Observation(qpos=motor_radians, image=frame))


####################################################################################################
# Program Entry Point
####################################################################################################

if __name__ == "__main__":
    set_seed(1)
    parser = argparse.ArgumentParser("act")

    # Train or infer
    parser.add_argument("--train", action="store_true", help="Train a model")
    parser.add_argument("--infer", action="store_true", help="Run inference server")

    # Training: file source and destination
    parser.add_argument("--checkpoint-dir", action="store", type=str, help="Directory to write checkpoints")
    parser.add_argument("--dataset-dir", action="store", type=str, help="Directories from which to read episodes (comma-separated, wildcards supported). Each must contain subdirectories with example episodes, e.g.: dataset_dir/example-*/data.hdf5.")

    # Inference: file source and server
    parser.add_argument("--checkpoint-file", action="store", type=str, help="Checkpoint filepath for inference (dataset_stats.pkl must be present in same directory)")
    parser.add_argument("--server-port", action="store", type=int, default=8001, help="Server port")

    # Training: parameters
    parser.add_argument("--policy-class", action="store", type=str, default="ACT", help="Policy class: ACT or CNNMLP")
    parser.add_argument("--batch-size", action="store", type=int, default=8, help="Batch size")
    parser.add_argument("--num-epochs", action="store", type=int, default=8000, help="Number of epochs to train for")
    parser.add_argument("--lr", action="store", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seed", action="store", type=int, default=42, help="Random seed")

    # Inference parameters
    parser.add_argument("--temporal-aggregation", action="store_true", help="Temporal aggregation")
    parser.add_argument("--num-cameras", action="store", type=int, default=1, help="Number of cameras in inference mode")

    # Training and inference: ACT
    parser.add_argument("--kl-weight", action="store", type=int, default=10, help="ACT model KL weight")
    parser.add_argument("--chunk-size", action="store", type=int, default=100, help="ACT model chunk size")
    parser.add_argument("--hidden-dim", action="store", type=int, default=512, help="ACT model hidden dimension")
    parser.add_argument("--dim-feedforward", action="store", type=int, default=32000, help="ACT model feed-forward dimension")

    # Validate
    options = parser.parse_args()
    if (not options.train and not options.infer) or (options.train and options.infer):
        parser.error("Please specify either --train or --infer")
    if options.train:
        if not options.checkpoint_dir:
            parser.error("--checkpoint-dir missing")
        if not options.dataset_dir:
            parser.error("--dataset-dir missing")
    if options.infer:
        if not options.checkpoint_file:
            parser.error("--checkpoint-file missing")

    # Run task
    if options.train:
        train(options=options)
    else:
        loop = asyncio.new_event_loop()
        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()
        tasks = []
        server = InferenceServer(port=options.server_port, input_queue=input_queue, output_queue=output_queue)
        tasks.append(loop.create_task(server.run()))
        tasks.append(loop.create_task(infer(options=options, input_queue=input_queue, output_queue=output_queue)))
        try:
            loop.run_until_complete(asyncio.gather(*tasks))
        except asyncio.exceptions.CancelledError:
            print("\nExited normally")
        except:
            print("\nExited due to uncaught exception")