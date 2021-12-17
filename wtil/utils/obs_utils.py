from typing import List, Tuple

from wtil.api.wt_pb2 import ObservationData, RLAIData


def split_obs(obs: ObservationData) -> Tuple[RLAIData, List[RLAIData]]:
    for i in range(len(obs.AIData)):
        if obs.AIData[i].EpisodeId == obs.EpisodeId:
            return obs.AIData[i], obs.AIData[:i] + obs.AIData[i:]
