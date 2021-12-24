import logging
import pathlib
from functools import partial
from typing import Any, Callable, List, Tuple

import torch
from google.protobuf.json_format import Parse

from wtil.api.wt_pb2 import ActionData, ObservationData, PlayerObservationData
from wtil.preprocess.act import encode_act
from wtil.preprocess.obs import encode_obs, process_obs
from wtil.utils.dataset import DirDataset, ProcessFn
from wtil.utils.logger import config_logger

ObsProcessor = Callable[[List[ObservationData]], List[Any]]
ObsEncoder = Callable[[List[Any], List[ActionData], int], Any]
ActEncoder = Callable[[List[Any], List[ActionData], int], Any]


def process_file(
    filename: str,
    obs_encoder: ObsEncoder,
    act_encoder: ActEncoder,
    *,
    obs_processer: ObsProcessor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    logging.debug(f"parse file: {filename}")
    with open(filename, "r") as f:
        obs_act_list = [Parse(line, PlayerObservationData()) for line in f]

    raw_obs_list = [obs_act.ObsData for obs_act in obs_act_list]
    raw_act_list = [obs_act.ActionDataArray[-1] for obs_act in obs_act_list]

    if obs_processer is not None:
        raw_obs_list = obs_processer(raw_obs_list)

    encoded_obs_list = [obs_encoder(raw_obs_list, raw_act_list, i) for i in range(len(raw_obs_list))]
    encoded_act_list = [act_encoder(raw_obs_list, raw_act_list, i) for i in range(len(raw_act_list))]

    encoded_obs_list = torch.tensor(encoded_obs_list, dtype=torch.float)
    encoded_act_list = torch.tensor(encoded_act_list, dtype=torch.float)
    return encoded_obs_list, encoded_act_list


def gen_process_fn(
    obs_encoder: ObsEncoder = encode_obs,
    act_encoder: ActEncoder = encode_act,
    obs_processer: ObsProcessor = process_obs,
) -> ProcessFn:

    return partial(
        process_file,
        obs_encoder=obs_encoder,
        act_encoder=act_encoder,
        obs_processer=obs_processer,
    )


def main():
    config_logger(False, False, None)

    data_dir = pathlib.Path(__file__).parent.parent / "data"

    path_glob = f"{data_dir}/*.txt"
    seq_len = 8
    batch_size = 16
    process_fn = gen_process_fn()
    num_workers = 1

    dir_dataloader = DirDataset.make_dataloader(path_glob, seq_len, batch_size, process_fn, num_workers)
    for file_dataloader in dir_dataloader:
        for file_data in file_dataloader:
            logging.info(f"processing file: {file_data.filename}")
            for batch in file_data:
                print(
                    "batch", type(batch), len(batch), batch[0].device, batch[0].shape, batch[1].device, batch[1].shape
                )


if __name__ == "__main__":
    main()
