#!/usr/bin/env python3

import argparse
import logging
import pathlib
from collections import defaultdict
from glob import glob

import numpy as np
from asyncenv.api.wt.wt_pb2 import PlayerObservationData, RLVector3D
from google.protobuf.json_format import Parse

from wtil.process.obs import process_obs
from wtil.utils.logger import config_logger

WORKING_DIR = pathlib.Path(__file__).parent.parent
DEFAULT_PATHNAME = str(WORKING_DIR / "data")
DEFAULT_FILEEXT = ".txt"

parser = argparse.ArgumentParser(
    description="wtil analyze utility",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--pathname", type=str, default=DEFAULT_PATHNAME, help="wt client action log location")
parser.add_argument("--fileext", type=str, default=DEFAULT_FILEEXT, help="action log file extension")


def has_value(dir: RLVector3D) -> int:
    v = np.linalg.norm([dir.X, dir.Y, dir.Z])
    return 0 if np.isclose(v, 0) else 1


def process_file(filename: str):

    logging.info(f"filename: {filename}")

    valid_act_len_stat = defaultdict(int)

    act_id_stat = defaultdict(int)
    act_len_stat = defaultdict(int)
    act_id_len_stat = defaultdict(int)
    act_id_with_move_dir = defaultdict(int)
    act_id_with_control_dir = defaultdict(int)

    record_num = 0

    with open(filename, "r") as f:
        for line in f:
            record_num += 1

            obs_act = Parse(line, PlayerObservationData())

            obs_data = obs_act.ObsData
            self_aidata, _ = process_obs(obs_data)
            valid_act_len_stat[len(self_aidata.ValidActions)] += 1

            act_list = obs_act.ActionDataArray
            act_len_stat[len(act_list)] += 1

            for act in act_list:
                act_id_len_stat[len(act.ActionID)] += 1

                has_move_dir = has_value(act.MoveDirection)
                has_control_dir = has_value(act.ControlDirection)

                for act_id in act.ActionID:
                    act_id_stat[act_id] += 1
                    act_id_with_move_dir[act_id] += has_move_dir
                    act_id_with_control_dir[act_id] += has_control_dir

    logging.info(f"record_num: {record_num}")

    logging.info(f"valid_act_len_stat: {valid_act_len_stat}")

    logging.info(f"act_id_stat: {act_id_stat}")
    logging.info(f"act_len_stat: {act_len_stat}")
    logging.info(f"act_id_len_stat: {act_id_len_stat}")
    logging.info(f"act_id_with_move_dir: {act_id_with_move_dir}")
    logging.info(f"act_id_with_control_dir: {act_id_with_control_dir}")


def main():

    config_logger(False, False, None)

    args = parser.parse_args()
    pattern = f"{args.pathname}/*{args.fileext}"

    filename_list = glob(pattern)
    logging.info(f"file_list: {filename_list}")

    if len(filename_list) == 0:
        logging.error(f"cannot find any file to process with pattern: {pattern}")

    for filename in filename_list:
        process_file(filename)


if __name__ == "__main__":
    main()
