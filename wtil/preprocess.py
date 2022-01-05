#!/usr/bin/env python3

import argparse
import logging
import os
import pathlib
from glob import glob
from typing import Any, Sequence, Tuple

import numpy as np
from asyncenv.api.wt.wt_pb2 import PlayerObservationData
from google.protobuf.json_format import Parse

from wtil.process.act import encode_act
from wtil.process.obs import encode_obs, process_obs
from wtil.utils.logger import config_logger

WORKING_DIR = pathlib.Path(__file__).parent.parent
DEFAULT_PATHNAME = str(WORKING_DIR / "data")
DEFAULT_FILEEXT = ".txt"
DEFAULT_OUTDIR = str(WORKING_DIR / "data_processed")
DEFAULT_NUM = 4096

parser = argparse.ArgumentParser(
    description="wtil preprocess utility",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--pathname", type=str, default=DEFAULT_PATHNAME, help="wt client action log location")
parser.add_argument("--fileext", type=str, default=DEFAULT_FILEEXT, help="action log file extension")
parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="where to store processed file")
parser.add_argument("--num", type=int, default=DEFAULT_NUM, help="number records per file")


def process_batch(outdir: str, basename: str, index: int, batch: Sequence[Tuple[Any, Any]]):

    logging.info(f"outdir: {outdir}, index:{index}, batch:{len(batch)}")

    obs_list = [t[0] for t in batch]
    act_list = [t[1] for t in batch]

    obs_list = [process_obs(o) for o in obs_list]

    encoded_obs_list = [encode_obs(o) for o in obs_list]
    encoded_act_list = [encode_act(obs_list[i - 1] if i > 0 else None, act_list[i]) for i in range(len(act_list))]

    obs_batch = {}
    for k in encoded_obs_list[0].keys():
        obs_batch[k] = np.stack([o[k] for o in encoded_obs_list])

    act_batch = {}
    for k in encoded_act_list[0].keys():
        act_batch[k] = np.stack([o[k] for o in encoded_act_list])

    data_batch = dict(obs=obs_batch, act=act_batch)

    filename = f"{outdir}/{basename}-{index}-{len(obs_list)}.npy"
    np.save(filename, data_batch)
    logging.info(f"save {filename} success")


def process_file(filename: str, outdir: str, num: int):

    logging.info(f"filename: {filename}, outdir:{outdir}")

    basename = os.path.splitext(os.path.basename(filename))[0]

    with open(filename, "r") as f:
        line_count = 0
        batch_index = 0
        batch = []

        for line in f:
            obs_act = Parse(line, PlayerObservationData())
            obs = obs_act.ObsData
            act = obs_act.ActionDataArray[-1]
            batch.append((obs, act))

            line_count += 1
            if num > 0 and line_count % num == 0:
                batch_index += 1
                process_batch(outdir, basename, batch_index, batch)
                batch = []

        if len(batch) > 0:
            batch_index += 1
            process_batch(outdir, basename, batch_index, batch)


def main():

    config_logger(False, False, None)

    args = parser.parse_args()
    pattern = f"{args.pathname}/*{args.fileext}"

    filename_list = glob(pattern)
    logging.info(f"file_list: {filename_list}")

    if len(filename_list) == 0:
        logging.error(f"cannot find any file to process with pattern: {pattern}")

    os.makedirs(args.outdir, exist_ok=True)

    for filename in filename_list:
        process_file(filename, args.outdir, args.num)


if __name__ == "__main__":
    main()
