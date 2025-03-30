from eoh.src.eoh import eoh
from eoh.src.eoh.utils.getParas import Paras
import random
import numpy as np

# Parameter initilization #
paras = Paras()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


set_seed(2025)

paras.set_paras(
    method="eoh",  # ['ael','eoh']
    problem="knapsack",  # ['tsp_construct','bp_online']
    llm_api_endpoint="dashscope.aliyuncs.com",  # set your LLM endpoint
    llm_api_key="sk-37dcd56707fd4cf6898ca516b76b3f88",  # set your key
    llm_model="qwen-max",
    ec_pop_size=4,  # number of samples in each population
    ec_n_pop=4,  # number of populations
    exp_n_proc=4,  # multi-core parallel
    exp_debug_mode=False,
)

# initilization
evolution = eoh.EVOL(paras)
evolution.run()
