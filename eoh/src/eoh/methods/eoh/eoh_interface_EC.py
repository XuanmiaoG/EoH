import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import numpy as np
import concurrent.futures


class InterfaceEC:
    def __init__(
        self,
        pop_size: int,
        m: int,
        api_endpoint: str | None,
        api_key: str | None,
        llm_model: str | None,
        llm_use_local: bool,
        llm_local_url: str | None,
        debug_mode: bool,
        interface_prob: object,
        select: object,
        n_p: int,
        timeout: float,
        use_numba: bool,
        **kwargs: object,
    ) -> None:
        """
        Interface for an Evolutionary Computation (EC) workflow.

        This class manages the generation and evaluation of offspring
        algorithms using a large language model (LLM)-based approach.

        Args:
            pop_size: Number of offspring (algorithms) to produce in each generation.
            m: Number of parents needed for certain evolutionary operators.
            api_endpoint: (Optional) The remote API endpoint for the LLM.
            api_key: (Optional) API key for remote LLM usage.
            llm_model: (Optional) The model name/type for the LLM.
            llm_use_local: If True, uses a local LLM server.
            llm_local_url: (Optional) URL for the local LLM server.
            debug_mode: If True, prints additional debug information.
            interface_prob: An object encapsulating problem-specific interfaces (e.g., TSP, Knapsack).
            select: An object defining how to select parents from the population.
            n_p: Number of parallel workers/processes to use.
            timeout: Time in seconds to wait for code evaluation before timeout.
            use_numba: If True, attempts to add Numba decorators to generated code.
            **kwargs: Any additional keyword arguments needed by the Evolution or the interface.

        Example math formula (illustrative):
            Let T be the total time (seconds) to evaluate one offspring code.
            Then for pop_size = P and concurrency = n_p,
            the approximate total evaluation time is:

                T_total ≈ (P / n_p) * T

            (neglecting overhead of scheduling/communication).
        """
        # LLM settings
        self.pop_size: int = pop_size
        self.m: int = m
        self.debug: bool = debug_mode
        self.n_p: int = n_p
        self.timeout: float = timeout
        self.use_numba: bool = use_numba

        self.interface_eval: object = (
            interface_prob  # Could type as a specific evaluator class
        )
        prompts = interface_prob.prompts

        # Initialize the Evolution object (assuming imported from .eoh_evolution)
        self.evol = Evolution(
            api_endpoint,
            api_key,
            llm_model,
            llm_use_local,
            llm_local_url,
            debug_mode,
            prompts,
            **kwargs,
        )

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select: object = select

    def code2file(self, code: str) -> None:
        """
        Write a string of code to a local file named 'ael_alg.py'.

        Args:
            code: The code string to be written to the file.
        """
        with open("./ael_alg.py", "w") as file:
            file.write(code)

    def add2pop(
        self, population: list[dict[str, object]], offspring: dict[str, object]
    ) -> bool:
        """
        Add an offspring to the population, avoiding duplicates by 'objective'.

        Args:
            population: Current list of individuals in the population.
            offspring: New individual to add.

        Returns:
            True if added successfully (no duplicate objectives), False otherwise.
        """
        for ind in population:
            if ind["objective"] == offspring["objective"]:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def check_duplicate(self, population: list[dict[str, object]], code: str) -> bool:
        """
        Check if the given code is already in the population.

        Args:
            population: The current list of individuals in the population.
            code: The code string to check.

        Returns:
            True if the code is found in the population, False otherwise.
        """
        for ind in population:
            if code == ind["code"]:
                return True
        return False

    def population_generation(self) -> list[dict[str, object]]:
        """
        Generate the initial population by using operator 'i1' to create individuals.

        Returns:
            A list of newly generated individuals.
        """
        n_create = 2
        population: list[dict[str, object]] = []

        for _ in range(n_create):
            _, pop = self.get_algorithm([], "i1")
            for p in pop:
                population.append(p)

        return population

    def population_generation_seed(
        self, seeds: list[dict[str, object]], n_p: int
    ) -> list[dict[str, object]]:
        """
        Initialize the population from a list of seed algorithms.

        Args:
            seeds: A list of seed algorithms, each containing 'code' and 'algorithm'.
            n_p: Number of parallel processes to use for evaluation.

        Returns:
            A list of individuals (with evaluated objectives).
        """
        population: list[dict[str, object]] = []

        fitness_results = Parallel(n_jobs=n_p)(
            delayed(self.interface_eval.evaluate)(seed["code"]) for seed in seeds
        )

        for i, seed in enumerate(seeds):
            try:
                seed_alg: dict[str, object] = {
                    "algorithm": seed["algorithm"],
                    "code": seed["code"],
                    "objective": None,
                    "other_inf": None,
                }
                obj = np.array(fitness_results[i])
                seed_alg["objective"] = np.round(obj, 5)
                population.append(seed_alg)
            except Exception:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")
        return population

    def _get_alg(
        self, pop: list[dict[str, object]], operator: str
    ) -> tuple[list[dict[str, object]] | None, dict[str, object]]:
        """
        Internal method to generate an offspring using a specified operator.

        Args:
            pop: Current population list.
            operator: A string indicating which evolutionary operator to use.

        Returns:
            (parents, offspring_dict)
            - parents: The parents used (or None if not applicable).
            - offspring_dict: A dict with fields 'algorithm', 'code', 'objective', 'other_inf'.
        """
        offspring = {
            "algorithm": None,
            "code": None,
            "objective": None,
            "other_inf": None,
        }

        parents = None
        if operator == "i1":
            [offspring["code"], offspring["algorithm"]] = self.evol.i1()

        elif operator == "e1":
            parents = self.select.parent_selection(pop, self.m)
            [offspring["code"], offspring["algorithm"]] = self.evol.e1(parents)

        elif operator == "e2":
            parents = self.select.parent_selection(pop, self.m)
            [offspring["code"], offspring["algorithm"]] = self.evol.e2(parents)

        elif operator == "m1":
            parents = self.select.parent_selection(pop, 1)
            [offspring["code"], offspring["algorithm"]] = self.evol.m1(parents[0])

        elif operator == "m2":
            parents = self.select.parent_selection(pop, 1)
            [offspring["code"], offspring["algorithm"]] = self.evol.m2(parents[0])

        elif operator == "m3":
            parents = self.select.parent_selection(pop, 1)
            [offspring["code"], offspring["algorithm"]] = self.evol.m3(parents[0])

        elif operator == "h1":
            parents = self.select.parent_selection(pop, self.m)
            [offspring["code"], offspring["algorithm"]] = self.evol.h1(parents)

        elif operator == "f1":
            parents = self.select.parent_selection(pop, 1)
            failure_cases = self.get_failure_cases(parents[0])
            [offspring["code"], offspring["algorithm"]] = self.evol.f1(
                parents[0], failure_cases
            )

        elif operator == "c1":
            parents = self.select.parent_selection(pop, 1)
            constraints = self.get_constraints()
            [offspring["code"], offspring["algorithm"]] = self.evol.c1(
                parents[0], constraints
            )

        elif operator == "p1":
            parents = self.select.parent_selection(pop, 1)
            complexity_target = self.get_complexity_target()
            [offspring["code"], offspring["algorithm"]] = self.evol.p1(
                parents[0], complexity_target
            )

        else:
            print(f"Evolution operator [{operator}] has not been implemented!\n")

        return parents, offspring

    def get_offspring(
        self, pop: list[dict[str, object]], operator: str
    ) -> tuple[list[dict[str, object]] | None, dict[str, object]]:
        """
        Generate a single offspring using 'operator', then evaluate it.

        - Avoids duplicates by checking existing population codes.
        - Optionally inserts a Numba decorator if use_numba=True.
        - Evaluates the newly generated code with a timeout.

        Args:
            pop: Current population list.
            operator: Name of the operator (e.g., 'm1', 'e1').

        Returns:
            (parents, offspring) where
            - parents: The parents selected for producing the offspring.
            - offspring: A dict with 'algorithm', 'code', 'objective', 'other_inf'.
                         If an error occurs, returns an empty structure.
        """
        try:
            p, offspring = self._get_alg(pop, operator)

            # Insert a Numba decorator if requested
            if self.use_numba:
                pattern = r"def\s+(\w+)\s*\(.*\):"
                match = re.search(pattern, offspring["code"])
                if match:
                    function_name = match.group(1)
                    code_modified = add_numba_decorator(
                        offspring["code"], function_name
                    )
                else:
                    code_modified = offspring["code"]
            else:
                code_modified = offspring["code"]

            n_retry = 1
            while self.check_duplicate(pop, offspring["code"]):
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                # Retry generation
                p, offspring = self._get_alg(pop, operator)
                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring["code"])
                    if match:
                        function_name = match.group(1)
                        code_modified = add_numba_decorator(
                            offspring["code"], function_name
                        )
                    else:
                        code_modified = offspring["code"]

                if n_retry > 1:
                    break

            # Evaluate the offspring with concurrency
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, code_modified)
                fitness = future.result(timeout=self.timeout)
                offspring["objective"] = np.round(fitness, 5)
                future.cancel()

        except Exception as e:
            if self.debug:
                print(f"Error in get_offspring: {e}")

            offspring = {
                "algorithm": None,
                "code": None,
                "objective": None,
                "other_inf": None,
            }
            p = None

        return p, offspring

    def get_algorithm(
        self, pop: list[dict[str, object]], operator: str
    ) -> tuple[list[list[dict[str, object]] | None], list[dict[str, object]]]:
        """
        Generate 'pop_size' offspring using the given operator in parallel,
        each with a separate job, then wait up to (timeout+15) seconds for all jobs.

        Args:
            pop: Current population.
            operator: The evolutionary operator to apply.

        Returns:
            (parents_list, offspring_list)
            - parents_list: A list of parents used for each offspring.
            - offspring_list: A list of offspring dictionaries.
        """
        results: list[tuple[list[dict[str, object]] | None, dict[str, object]]] = []
        try:
            results = Parallel(n_jobs=self.n_p, timeout=self.timeout + 15)(
                delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size)
            )
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out .")

        time.sleep(2)

        out_p: list[list[dict[str, object]] | None] = []
        out_off: list[dict[str, object]] = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offspring: \n {off}")

        return out_p, out_off

    # Example stubs for optional operators (f1, c1, p1):
    def get_failure_cases(self, parent: dict[str, object]) -> list[object]:
        """
        Stub function to retrieve failure cases for 'f1' operator.
        """
        return []

    def get_constraints(self) -> dict[str, object]:
        """
        Stub function to retrieve constraints for 'c1' operator.
        """
        return {}

    def get_complexity_target(self) -> float:
        """
        Stub function to retrieve complexity target for 'p1' operator.
        """
        return 0.0
