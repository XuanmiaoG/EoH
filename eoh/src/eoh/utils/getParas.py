class Paras:
    def __init__(self) -> None:
        """
        Class holding various parameters for an evolutionary or learning method.

        Attributes:
            method (str): The method name (e.g., 'eoh', 'ls', 'sa', 'ael').
            problem (str): The problem type (e.g., 'tsp_construct', 'bp_online').
            selection (str | None): The selection procedure for the population.
            management (str | None): The population management strategy.
            ec_pop_size (int): Number of algorithms in each population.
            ec_n_pop (int): Number of populations.
            ec_operators (list[str] | None): The operators (e.g., crossover, mutation).
            ec_m (int): Number of parents for certain operators (e.g., 'e1', 'e2').
            ec_operator_weights (list[float] | None): Probabilities of choosing each operator.
            llm_use_local (bool): If True, use a local large language model server.
            llm_local_url (str | None): URL of the local LLM server.
            llm_api_endpoint (str | None): Endpoint for a remote LLM.
            llm_api_key (str | None): API key for a remote LLM.
            llm_model (str | None): Which model type to use for a remote LLM.
            exp_debug_mode (bool): If True, enables debug mode.
            exp_output_path (str): Folder path for outputs.
            exp_use_seed (bool): If True, seeds are used for reproducibility.
            exp_seed_path (str): File path to a JSON with seeds.
            exp_use_continue (bool): If True, continue from a previous run.
            exp_continue_id (int): ID of the continue-run scenario.
            exp_continue_path (str): File path for a population JSON if continuing.
            exp_n_proc (int): Number of processes to use. If -1, use all CPU cores.
            eva_timeout (int): Timeout (seconds) for evaluations.
            eva_numba_decorator (bool): If True, applies Numba JIT compilation where applicable.
        """
        #####################
        # General settings
        #####################
        self.method: str = "eoh"
        self.problem: str = "tsp_construct"
        self.selection: str | None = None
        self.management: str | None = None

        #####################
        # EC settings
        #####################
        self.ec_pop_size: int = 5
        self.ec_n_pop: int = 5
        self.ec_operators: list[str] | None = None
        self.ec_m: int = 2
        self.ec_operator_weights: list[float] | None = None

        #####################
        # LLM settings
        #####################
        self.llm_use_local: bool = False
        self.llm_local_url: str | None = None
        self.llm_api_endpoint: str | None = None
        self.llm_api_key: str | None = None
        self.llm_model: str | None = None

        #####################
        # Exp settings
        #####################
        self.exp_debug_mode: bool = False
        self.exp_output_path: str = "./"
        self.exp_use_seed: bool = False
        self.exp_seed_path: str = "./seeds/seeds.json"
        self.exp_use_continue: bool = False
        self.exp_continue_id: int = 0
        self.exp_continue_path: str = "./results/pops/population_generation_0.json"
        self.exp_n_proc: int = 1

        #####################
        # Evaluation settings
        #####################
        self.eva_timeout: int = 30
        self.eva_numba_decorator: bool = False

    def set_parallel(self) -> None:
        """
        Set the number of processes for parallelization by comparing
        'exp_n_proc' with the actual CPU count.

        Example math formula (for demonstration):
            Let p = min(exp_n_proc, cpu_count).
            If exp_n_proc == -1, we choose p = cpu_count.

            p = exp_n_proc, if 0 < exp_n_proc <= cpu_count
              = cpu_count,  if exp_n_proc == -1 or exp_n_proc > cpu_count
        """
        import multiprocessing

        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")

    def set_ec(self) -> None:
        """
        Configure evolution-related parameters such as management, selection, and operators.

        - The method defines which operators or management strategy are set by default.
        - If ec_operator_weights is not provided or has a length mismatch,
          it resets to uniform [1, 1, ..., 1].
        - For single-point-based methods, sets ec_pop_size to 1.
        """
        # Set default management
        if self.management is None:
            if self.method in ["ael", "eoh"]:
                self.management = "pop_greedy"
            elif self.method == "ls":
                self.management = "ls_greedy"
            elif self.method == "sa":
                self.management = "ls_sa"

        # Set default selection
        if self.selection is None:
            self.selection = "prob_rank"

        # Set default operators
        if self.ec_operators is None:
            if self.method == "eoh":
                self.ec_operators = ["e1", "e2", "m1", "m2"]
            elif self.method == "ael":
                self.ec_operators = ["crossover", "mutation"]
            elif self.method == "ls":
                self.ec_operators = ["m1"]
            elif self.method == "sa":
                self.ec_operators = ["m1"]
            elif self.method == "mls":
                self.ec_operators = ["m1", "m2"]

        # Set operator weights
        if self.ec_operator_weights is None:
            self.ec_operator_weights = [1.0 for _ in range(len(self.ec_operators))]
        elif len(self.ec_operators) != len(self.ec_operator_weights):
            print(
                "Warning! Lengths of ec_operator_weights and ec_operator should be the same."
            )
            self.ec_operator_weights = [1.0 for _ in range(len(self.ec_operators))]

        # For single-point-based methods, force pop_size=1
        if self.method in ["ls", "sa"] and self.ec_pop_size > 1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1. ")

    def set_evaluation(self) -> None:
        """
        Initialize evaluation settings based on the problem type.

        If problem == 'bp_online', modifies eva_timeout and applies numba decorator.
        If problem == 'tsp_construct', sets eva_timeout = 20.
        """
        if self.problem == "bp_online":
            self.eva_timeout = 20
            self.eva_numba_decorator = True
        elif self.problem == "tsp_construct":
            self.eva_timeout = 20

    def set_paras(self, *args: object, **kwargs: object) -> None:
        """
        Dynamically update parameters using the kwargs provided.
        Then apply method-specific and problem-specific configurations.

        :param args: Additional positional arguments (not used here).
        :param kwargs: Key-value pairs corresponding to attributes in this class.
        """
        # Update any attributes that match kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Set parallelization
        self.set_parallel()

        # Initialize method and EC settings
        self.set_ec()

        # Initialize evaluation settings
        self.set_evaluation()


if __name__ == "__main__":
    # Create an instance of the Paras class
    paras_instance = Paras()

    # Setting parameters using the set_paras method
    paras_instance.set_paras(
        llm_use_local=True, llm_local_url="http://example.com", ec_pop_size=8
    )

    # Accessing the updated parameters
    print(paras_instance.llm_use_local)  # Output: True
    print(paras_instance.llm_local_url)  # Output: http://example.com
    print(paras_instance.ec_pop_size)  # Output: 8
