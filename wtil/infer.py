#!/usr/bin/env python3
import logging
import random
import time
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
from asyncenv.api.wt.wt_pb2 import ActionData, ActionDatas, ObservationData, ObservationDatas
from asyncenv.client.wt.env_proxy import EnvProxy
from asyncenv.test.wt.utils.env_loop import wt_env_loop

from wtil.model import Model
from wtil.process.act import decode_act
from wtil.process.obs import encode_obs, process_obs
from wtil.utils.logger import config_logger

USE_CUDA = torch.cuda.is_available()
USE_DEVICE = "cuda" if USE_CUDA else "cpu"

MODEL = None
PTH = "/home/k/repos/wtil/checkpoints/wtil_0.97461.pth"

SEQ_LEN = 30
PREV_PROCESSED_OBS_LIST = None
CURR_PROCESSED_OBS_LIST = None
ENCODED_OBS_LIST_DEQUE = deque(maxlen=SEQ_LEN)


def to_torch(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).type(torch.float).to(USE_DEVICE) for k, v in data.items()}


def add_obs_list(obs_list: ObservationDatas):
    global PREV_PROCESSED_OBS_LIST, CURR_PROCESSED_OBS_LIST, ENCODED_OBS_LIST_DEQUE

    PREV_PROCESSED_OBS_LIST = CURR_PROCESSED_OBS_LIST
    CURR_PROCESSED_OBS_LIST = []
    for i in range(len(obs_list.ObservationDataArray)):
        obs: ObservationData = obs_list.ObservationDataArray[i]
        processed_obs = process_obs(obs)
        CURR_PROCESSED_OBS_LIST.append(processed_obs)

    ENCODED_OBS_LIST_DEQUE.append([to_torch(encode_obs(obs)) for obs in CURR_PROCESSED_OBS_LIST])


def get_obs_batch() -> Optional[Dict[str, torch.Tensor]]:
    if len(ENCODED_OBS_LIST_DEQUE) < SEQ_LEN:
        return None

    agent_num = len(ENCODED_OBS_LIST_DEQUE[0])
    obs_batch = {}
    for k in ENCODED_OBS_LIST_DEQUE[0][0].keys():
        # obs_batch[k] = np.stack([np.stack([o[i][k] for o in ENCODED_OBS_LIST_DEQUE]) for i in range(agent_num)])
        obs_batch[k] = torch.stack([torch.stack([o[i][k] for o in ENCODED_OBS_LIST_DEQUE]) for i in range(agent_num)])

    return obs_batch


def get_action(obs_list: ObservationDatas, extra_info) -> ActionDatas:
    add_obs_list(obs_list)

    obs_batch = get_obs_batch()
    if obs_batch is None:
        action_list = [ActionData() for _ in range(len(obs_list.ObservationDataArray))]
    else:
        with torch.no_grad():
            batched_predict = MODEL(obs_batch)
            batched_predict = {k: v.cpu().numpy() for k, v in batched_predict.items()}
        action_list = []
        for i in range(len(PREV_PROCESSED_OBS_LIST)):
            data = {k: batched_predict[k][i] for k in batched_predict}
            action_list.append(decode_act(PREV_PROCESSED_OBS_LIST[i], data))

    return ActionDatas(ActionDataArray=action_list)


def load_model():
    global MODEL

    MODEL = Model()
    if len(PTH) > 0:
        logging.info(f"load model from file: {PTH}")
        MODEL.load_state_dict(torch.load(PTH))
    MODEL.to(USE_DEVICE)
    MODEL.train(False)


def main(server: Optional[str] = "localhost:9091"):

    config_logger(False, False, None)

    load_model()

    while True:
        random.seed(time.time())
        env = EnvProxy.wait_envs(
            server_addr=server,  # grpc server address
            env_num=1,  # environment number
            interval=1,  # sleep interval (second) for wait environment get ready
            reset_timeout=10,  # reset action timeout
            step_timeout=10,  # step action timeout
            msg_to_dict=False,
        )[0]

        wt_env_loop(env, get_action, None, 1000, False, False)


if __name__ == "__main__":
    main()
