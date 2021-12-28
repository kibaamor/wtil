#!/usr/bin/env python3
import random
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
from asyncenv.api.wt.wt_pb2 import ActionData, ActionDatas, ObservationData, ObservationDatas
from asyncenv.client.wt.env_proxy import EnvProxy
from asyncenv.test.wt.utils.env_loop import wt_env_loop

from wtil.model import Model
from wtil.process.act import decode_act
from wtil.process.obs import encode_obs, process_obs
from wtil.utils.logger import config_logger

USE_CUDA = False  # torch.cuda.is_available()
USE_DEVICE = "cuda" if USE_CUDA else "cpu"

MODEL = None
PTH = ""

SEQ_LEN = 30
PREV_PROCESSED_OBS_LIST = None
CURR_PROCESSED_OBS_LIST = None
ENCODED_OBS_LIST_DEQUE = deque(maxlen=SEQ_LEN)


def add_obs_list(obs_list: ObservationDatas):
    global PREV_PROCESSED_OBS_LIST, CURR_PROCESSED_OBS_LIST, ENCODED_OBS_LIST_DEQUE

    PREV_PROCESSED_OBS_LIST = CURR_PROCESSED_OBS_LIST
    CURR_PROCESSED_OBS_LIST = []
    for i in range(len(obs_list.ObservationDataArray)):
        obs: ObservationData = obs_list.ObservationDataArray[i]
        processed_obs = process_obs(obs)
        CURR_PROCESSED_OBS_LIST.append(processed_obs)

    ENCODED_OBS_LIST_DEQUE.append([encode_obs(obs) for obs in CURR_PROCESSED_OBS_LIST])


def get_batched_obs() -> Optional[np.ndarray]:
    if len(ENCODED_OBS_LIST_DEQUE) < SEQ_LEN:
        return None

    batched_obs = np.array(ENCODED_OBS_LIST_DEQUE)
    batched_obs = batched_obs.swapaxes(0, 1)
    return batched_obs


def get_action(obs_list: ObservationDatas, extra_info) -> ActionDatas:
    add_obs_list(obs_list)

    batched_obs = get_batched_obs()
    if batched_obs is None:
        action_list = [ActionData() for _ in range(len(obs_list.ObservationDataArray))]
    else:
        with torch.no_grad():
            batched_obs = torch.tensor(batched_obs, dtype=torch.float, device=USE_DEVICE)
            batched_predict = MODEL(batched_obs)
            batched_predict = batched_predict.cpu().numpy()
        action_list = []
        for i in range(len(PREV_PROCESSED_OBS_LIST)):
            action_list.append(decode_act(PREV_PROCESSED_OBS_LIST[i], batched_predict[i]))

    return ActionDatas(ActionDataArray=action_list)


def load_model():
    global MODEL

    MODEL = Model().to(USE_DEVICE)
    if len(PTH) > 0:
        MODEL.load_state_dict(torch.load(PTH))
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

        wt_env_loop(env, get_action, None, 1000, False)


if __name__ == "__main__":
    main()
