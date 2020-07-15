#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import deep_sdf
import deep_sdf.workspace as ws
from tensorboardX import SummaryWriter

def getWriterPath(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return prefix + task + '/' + exper_name + str_date_time

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def tb_scalar_dict(writer, losses, task="training"):
    """
    # add scalar dictionary to tensorboard
    :param losses:
    :param task:
    :return:
    """
    for element in list(losses):
        writer.add_scalar(task + "-" + element, losses[element], self.n_iter)


def load_logs(experiment_directory, type):

    writer = SummaryWriter(getWriterPath(task='.', 
        exper_name=experiment_directory, date=True))
    logs = torch.load(os.path.join(experiment_directory, ws.logs_filename))

    logging.info("latest epoch is {}".format(logs["epoch"]))

    num_iters = len(logs["loss"])
    iters_per_epoch = num_iters / logs["epoch"]

    logging.info("{} iters per epoch".format(iters_per_epoch))

    smoothed_loss_41 = running_mean(logs["loss"], 41)
    smoothed_loss_1601 = running_mean(logs["loss"], 1601)

    # fig, ax = plt.subplots()

    for i, l in enumerate(logs["loss"]):
        writer.add_scalar('training-loss', l, i)

    # if type == "loss":

    #     ax.plot(
    #         np.arange(num_iters) / iters_per_epoch,
    #         logs["loss"],
    #         "#82c6eb",
    #         np.arange(20, num_iters - 20) / iters_per_epoch,
    #         smoothed_loss_41,
    #         "#2a9edd",
    #         np.arange(800, num_iters - 800) / iters_per_epoch,
    #         smoothed_loss_1601,
    #         "#16628b",
    #     )

    #     ax.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")

    combined_lrs = np.array(logs["learning_rate"])
    for i, l in enumerate(combined_lrs):
        writer.add_scalar('training-lr-0', l[0], i)
        writer.add_scalar('training-lr-1', l[1], i)

    # elif type == "learning_rate":
    #     combined_lrs = np.array(logs["learning_rate"])

    #     ax.plot(
    #         np.arange(combined_lrs.shape[0]),
    #         combined_lrs[:, 0],
    #         np.arange(combined_lrs.shape[0]),
    #         combined_lrs[:, 1],
    #     )
    #     ax.set(xlabel="Epoch", ylabel="Learning Rate", title="Learning Rates")

    # elif type == "time":
    #     ax.plot(logs["timing"], "#833eb7")
    #     ax.set(xlabel="Epoch", ylabel="Time per Epoch (s)", title="Timing")


    for i, l in enumerate(logs["latent_magnitude"]):
        writer.add_scalar('training-latent_magnitude', l, i)

    # for i, l in enumerate(logs["param_magnitude"]):
    #     writer.add_scalar('training-param_magnitude', l[], i)


    # elif type == "lat_mag":
    #     ax.plot(logs["latent_magnitude"])
    #     ax.set(xlabel="Epoch", ylabel="Magnitude", title="Latent Vector Magnitude")

    # elif type == "param_mag":
    #     for _name, mags in logs["param_magnitude"].items():
    #         ax.plot(mags)
    #     ax.set(xlabel="Epoch", ylabel="Magnitude", title="Parameter Magnitude")
    #     ax.legend(logs["param_magnitude"].keys())

    # else:
    #     raise Exception('unrecognized plot type "{}"'.format(type))

    # ax.grid()
    # plt.show()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Plot DeepSDF training logs")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment "
        + "specifications in 'specs.json', and logging will be done in this directory "
        + "as well",
    )
    arg_parser.add_argument("--type", "-t", dest="type", default="loss")

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    load_logs(args.experiment_directory, args.type)
