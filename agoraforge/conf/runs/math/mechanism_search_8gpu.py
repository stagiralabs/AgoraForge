"""Eight-GPU SHADE learned-mechanism search.

N=19 gives 38 parent/trial candidates after initialization. The two configured
probes make 40 evaluations, dispatched as eight chunks of five. Generation 0 has
21 evaluations and therefore ends with a partial wave.
"""
from agoraforge.conf.runs.math.mechanism_search import get_config as _base


def get_config():
    cfg = _base()
    s = cfg.search
    s.population = 19
    s.batched_chunk = 5
    s.gpus = "0,1,2,3,4,5,6,7"
    s.threads_per_worker = 1
    s.amp_dtype = "bf16"
    s.crn = True                 # common random numbers cut candidate-ranking noise
    s.wandb_mode = "online"
    s.wandb_tags = ["mechanism-search", "shade", "8gpu", "cold-start"]
    return cfg
