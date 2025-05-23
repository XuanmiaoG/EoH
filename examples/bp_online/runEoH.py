from eoh import eoh
from eoh.utils.getParas import Paras

# Parameter initilization #
paras = Paras()

# Set parameters #
paras.set_paras(
    method="eoh",  # ['ael','eoh']
    problem="bp_online",  # ['tsp_construct','bp_online']
    llm_api_endpoint="api.deepseek.com",  # set your LLM endpoint
    llm_api_key="sk-35e4e0944b4b495e9bfb670d7198d83b",  # set your key
    llm_model="deepseek-chat",
    ec_pop_size=4,  # number of samples in each population
    ec_n_pop=4,  # number of populations
    exp_n_proc=4,  # multi-core parallel
    exp_debug_mode=False,  # debug mode
)

# initilization
evolution = eoh.EVOL(paras)

# run
evolution.run()
