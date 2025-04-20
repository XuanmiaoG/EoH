import re
import time
import json
import random
import numpy as np
from .eoh_interface_EC import InterfaceEC  # Adjust your import path as needed


class EOH:
    """
    The EOH (Evolution of Heuristics) class manages an evolutionary loop:
    1) Manages a population of algorithms (individuals).
    2) Uses an InterfaceEC to produce new offspring from parents (via LLM).
    3) Maintains and evolves the population over multiple generations.

    Strict Thought:
      - We do not import 'typing', only Python 3.10 union syntax (Type | None).
      - We comment the code to clarify parameters and flow.
    """

    def __init__(
        self,
        paras: object,  # The object holding user-defined parameters
        problem: object,  # The problem interface (for evaluation)
        select: object,  # The object handling selection methods
        manage: object,  # The object handling population management
        **kwargs: object,
    ) -> None:
        """
        Constructor for EOH. Loads parameters from 'paras' and sets up the evolutionary environment.

        Args:
            paras: Contains user-defined configuration (pop sizes, LLM settings, etc.).
            problem: The problem interface (evaluation logic).
            select: The selection object (which might handle parent selection).
            manage: The population management object (which might handle how to keep best individuals).
            **kwargs: Additional parameters (not used in current code).
        """
        self.prob: object = problem
        self.select: object = select
        self.manage: object = manage

        # LLM settings
        self.use_local_llm: bool = paras.llm_use_local
        self.llm_local_url: str | None = paras.llm_local_url
        self.api_endpoint: str | None = paras.llm_api_endpoint
        self.api_key: str | None = paras.llm_api_key
        self.llm_model: str | None = paras.llm_model

        # Evolutionary settings
        self.pop_size: int = paras.ec_pop_size  # population size
        self.n_pop: int = paras.ec_n_pop  # number of generations (populations)
        self.operators: list[str] = paras.ec_operators  # the operator names
        self.operator_weights: list[float] = (
            paras.ec_operator_weights
        )  # each operator's probability
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print(
                "m should not be larger than pop size or smaller than 2, adjust it to m=2"
            )
            paras.ec_m = 2
        self.m: int = paras.ec_m  # number of parents for certain operators

        self.debug_mode: bool = paras.exp_debug_mode
        self.ndelay: int = 1  # a default delay parameter (not used in code)

        # Seeding / population load
        self.use_seed: bool = paras.exp_use_seed
        self.seed_path: str = paras.exp_seed_path
        self.load_pop: bool = paras.exp_use_continue
        self.load_pop_path: str = paras.exp_continue_path
        self.load_pop_id: int = paras.exp_continue_id

        # Output / experiment
        self.output_path: str = paras.exp_output_path
        self.exp_n_proc: int = paras.exp_n_proc
        self.timeout: int = paras.eva_timeout
        self.use_numba: bool = paras.eva_numba_decorator

        print("- EoH parameters loaded -")

        # We set a random seed for reproducibility
        random.seed(2024)

    def add2pop(self, population: list[dict], offspring: list[dict]) -> None:
        """
        Attempts to add new offspring individuals to 'population'. If duplicates found
        (by objective), it prints a debug message (when debug_mode = True).

        Args:
            population: The current population (list of dicts).
            offspring: Newly created individuals to be added.
        """
        for off in offspring:
            for ind in population:
                if ind["objective"] == off["objective"]:
                    if self.debug_mode:
                        print("duplicated result, retrying ... ")
            population.append(off)

    def run(self) -> None:
        """
        Main evolutionary loop:
         1) Creates or loads an initial population.
         2) Iterates over 'self.n_pop' generations.
         3) Each generation tries each operator, obtains offspring, and manages the population.
         4) Saves population states to files.
        """
        print("- Evolution Start -")
        time_start: float = time.time()

        # We create an interface to the problem (evaluation) or get it from self.prob
        interface_prob = self.prob
        interface_ec = InterfaceEC(
            pop_size=self.pop_size,
            m=self.m,
            api_endpoint=self.api_endpoint,
            api_key=self.api_key,
            llm_model=self.llm_model,
            llm_use_local=self.use_local_llm,
            llm_local_url=self.llm_local_url,
            debug_mode=self.debug_mode,
            interface_prob=interface_prob,
            select=self.select,
            n_p=self.exp_n_proc,
            timeout=self.timeout + 1000,
            use_numba=self.use_numba,
        )

        # Initialization
        population: list[dict] = []
        if self.use_seed:
            # Load seeds from file
            with open(self.seed_path, "r") as file:
                data = json.load(file)
            # population_generation_seed presumably does an evaluation of those seeds
            population = interface_ec.population_generation_seed(data, self.exp_n_proc)
            filename_seed: str = (
                self.output_path + "/results/pops/population_generation_0.json"
            )
            with open(filename_seed, "w") as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            # Not using seed
            if self.load_pop:
                # load population from a file
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path, "r") as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:
                # create a new population using the interface_ec
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(
                    population, self.pop_size
                )

                print("Pop initial: ")
                for off in population:
                    print(" Obj: ", off["objective"], end="|")
                print()
                print("initial population has been created!")

                # Save the population to file
                filename_init: str = (
                    self.output_path + "/results/pops/population_generation_0.json"
                )
                with open(filename_init, "w") as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        # Main evolutionary loop over populations
        n_op: int = len(self.operators)

        for pop_i in range(n_start, self.n_pop):
            # For each generation, we apply each operator
            for i in range(n_op):
                op: str = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                op_w: float = self.operator_weights[i]

                if np.random.rand() < op_w:
                    # get_algorithm presumably returns a (parents, offspring) tuple
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                    # add offsprings to population (checking duplication is minimal though)
                    self.add2pop(population, offsprings)

                    for off in offsprings:
                        print(" Obj: ", off["objective"], end="|")

                    # Then manage the population (truncate or keep best, etc.)
                    size_act: int = min(len(population), self.pop_size)
                    population = self.manage.population_management(population, size_act)
                print()

            # Save population after this generation
            filename_pop: str = (
                f"{self.output_path}/results/pops/population_generation_{pop_i + 1}.json"
            )
            with open(filename_pop, "w") as f:
                json.dump(population, f, indent=5)

            # Save the best one (or entire pop[0]) to a separate file
            filename_best: str = (
                f"{self.output_path}/results/pops_best/population_generation_{pop_i + 1}.json"
            )
            with open(filename_best, "w") as f:
                json.dump(population[0], f, indent=5)

            # Print summary info
            time_spent: float = (time.time() - time_start) / 60.0
            print(
                f"--- {pop_i + 1} of {self.n_pop} populations finished. Time Cost: {time_spent:.1f} m"
            )
            print("Pop Objs: ", end=" ")
            for ind in population:
                print(str(ind["objective"]) + " ", end="")
            print()
