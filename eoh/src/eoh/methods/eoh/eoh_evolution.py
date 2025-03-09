import re
import time
from ...llm.interface_LLM import InterfaceLLM


class Evolution:
    def __init__(
        self,
        api_endpoint: str | None,
        api_key: str | None,
        model_LLM: str | None,
        llm_use_local: bool,
        llm_local_url: str | None,
        debug_mode: bool,
        prompts: object,
        **kwargs: object,
    ) -> None:

        # Retrieve prompt info
        self.prompt_task: str = prompts.get_task()
        self.prompt_func_name: str = prompts.get_func_name()
        self.prompt_func_inputs: list[str] = prompts.get_func_inputs()
        self.prompt_func_outputs: list[str] = prompts.get_func_outputs()
        self.prompt_inout_inf: str = prompts.get_inout_inf()
        self.prompt_other_inf: str = prompts.get_other_inf()

        # Join inputs/outputs with commas if more than one
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs: str = ", ".join(
                "'" + s + "'" for s in self.prompt_func_inputs
            )
        else:
            self.joined_inputs: str = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs: str = ", ".join(
                "'" + s + "'" for s in self.prompt_func_outputs
            )
        else:
            self.joined_outputs: str = "'" + self.prompt_func_outputs[0] + "'"

        self.api_endpoint: str | None = api_endpoint
        self.api_key: str | None = api_key
        self.model_LLM: str | None = model_LLM
        self.debug_mode: bool = debug_mode

        self.interface_llm = InterfaceLLM(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            llm_use_local,
            llm_local_url,
            self.debug_mode,
        )

    def get_prompt_i1(self) -> str:
        prompt_content = (
            self.prompt_task
            + "\n"
            + "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_e1(self, indivs: list[dict[str, str]]) -> str:
        prompt_indiv = ""
        for i, indiv in enumerate(indivs):
            prompt_indiv += (
                f"No.{i + 1} algorithm and the corresponding code are: \n"
                + indiv["algorithm"]
                + "\n"
                + indiv["code"]
                + "\n"
            )

        prompt_content = (
            self.prompt_task
            + "\n"
            + "I have "
            + str(len(indivs))
            + " existing algorithms with their codes as follows: \n"
            + prompt_indiv
            + "Please help me create a new algorithm that has a totally different form from the given ones. \n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_e2(self, indivs: list[dict[str, str]]) -> str:
        prompt_indiv = ""
        for i, indiv in enumerate(indivs):
            prompt_indiv += (
                f"No.{i + 1} algorithm and the corresponding code are: \n"
                + indiv["algorithm"]
                + "\n"
                + indiv["code"]
                + "\n"
            )

        prompt_content = (
            self.prompt_task
            + "\n"
            + "I have "
            + str(len(indivs))
            + " existing algorithms with their codes as follows: \n"
            + prompt_indiv
            + "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"
            "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. "
            "The description must be inside a brace. Thirdly, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_m1(self, indiv1: dict[str, str]) -> str:
        prompt_content = (
            self.prompt_task
            + "\n"
            + "I have one algorithm with its code as follows. Algorithm description: "
            + indiv1["algorithm"]
            + "\nCode:\n"
            + indiv1["code"]
            + "\nPlease assist me in creating a new algorithm that has a different form "
            "but can be a modified version of the algorithm provided. \n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_m2(self, indiv1: dict[str, str]) -> str:
        prompt_content = (
            self.prompt_task
            + "\n"
            + "I have one algorithm with its code as follows. Algorithm description: "
            + indiv1["algorithm"]
            + "\nCode:\n"
            + indiv1["code"]
            + "\nPlease identify the main algorithm parameters and assist me in creating a new algorithm "
            "that has a different parameter settings of the score function provided. \n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_m3(self, indiv1: dict[str, str]) -> str:
        prompt_content = (
            "First, you need to identify the main components in the function below. "
            "Next, analyze whether any of these components can be overfit to the in-distribution instances. "
            "Then, based on your analysis, simplify the components to enhance the generalization "
            "to potential out-of-distribution instances. Finally, provide the revised code, "
            "keeping the function name, inputs, and outputs unchanged. \n"
            + indiv1["code"]
            + "\n"
            + self.prompt_inout_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_h1(self, indivs: list[dict[str, str]]) -> str:
        prompt_indiv = ""
        for i, indiv in enumerate(indivs):
            prompt_indiv += (
                f"No.{i + 1} algorithm and the corresponding code are: \n"
                + indiv["algorithm"]
                + "\n"
                + indiv["code"]
                + "\n"
            )

        prompt_content = (
            self.prompt_task
            + "\n"
            + "I have "
            + str(len(indivs))
            + " existing algorithms with their codes as follows: \n"
            + prompt_indiv
            + "Please create a new hybrid algorithm by:\n"
            "1. Identifying the unique strengths of each algorithm\n"
            "2. Combining their complementary features\n"
            "3. Adding novel elements to improve the combination\n\n"
            "First, describe your hybrid algorithm and its integration approach in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_f1(self, indiv1: dict[str, str], failure_cases: str) -> str:
        prompt_content = (
            self.prompt_task
            + "\n"
            + "I have one algorithm with its code as follows. Algorithm description: "
            + indiv1["algorithm"]
            + "\nCode:\n"
            + indiv1["code"]
            + "\nFailure cases:\n"
            + failure_cases
            + "\n\n"
            "Please create an improved algorithm by:\n"
            "1. Analyzing the patterns in failure cases\n"
            "2. Identifying the root causes of poor performance\n"
            "3. Designing specific mechanisms to address these weaknesses\n\n"
            "First, describe your improved algorithm in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_c1(self, indiv1: dict[str, str], constraints: str) -> str:
        prompt_content = (
            self.prompt_task
            + "\n"
            + "I have one algorithm with its code as follows. Algorithm description: "
            + indiv1["algorithm"]
            + "\nCode:\n"
            + indiv1["code"]
            + "\nCritical constraints to address:\n"
            + constraints
            + "\n\n"
            "Please create a constraint-optimized algorithm by:\n"
            "1. Analyzing how the current algorithm handles these constraints\n"
            "2. Designing specialized mechanisms for constraint satisfaction\n"
            "3. Balancing constraint handling with overall performance\n\n"
            "First, describe your constraint-focused algorithm in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_p1(self, indiv1: dict[str, str], complexity_target: str) -> str:
        prompt_content = (
            self.prompt_task
            + "\n"
            + "I have one algorithm with its code as follows. Algorithm description: "
            + indiv1["algorithm"]
            + "\nCode:\n"
            + indiv1["code"]
            + "\nTarget complexity aspects:\n"
            + complexity_target
            + "\n\n"
            "Please create an enhanced algorithm by:\n"
            "1. Starting with the core mechanism of the base algorithm\n"
            "2. Adding complexity layers that address: " + complexity_target + "\n"
            "3. Maintaining interpretability and efficiency\n\n"
            "First, describe your progressively enhanced algorithm in one sentence. "
            "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )
        return prompt_content

    ############################################################################
    # Core Response Handling
    ############################################################################

    def _get_alg(self, prompt_content: str) -> list[str]:
        response: str = self.interface_llm.get_response(prompt_content)

        # Attempt to extract the algorithm description (anything within curly braces)
        algorithm_matches = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm_matches) == 0:
            # fallback checks
            if "python" in response:
                algorithm_matches = re.findall(r"^.*?(?=python)", response, re.DOTALL)
            elif "import" in response:
                algorithm_matches = re.findall(r"^.*?(?=import)", response, re.DOTALL)
            else:
                algorithm_matches = re.findall(r"^.*?(?=def)", response, re.DOTALL)

        # Attempt to extract the code snippet
        code_matches = re.findall(r"import.*return", response, re.DOTALL)
        if len(code_matches) == 0:
            code_matches = re.findall(r"def.*return", response, re.DOTALL)

        n_retry: int = 1
        while len(algorithm_matches) == 0 or len(code_matches) == 0:
            if self.debug_mode:
                print(
                    "Error: algorithm or code not identified, wait 1s and retrying ... "
                )
            time.sleep(1)

            response = self.interface_llm.get_response(prompt_content)
            algorithm_matches = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm_matches) == 0:
                if "python" in response:
                    algorithm_matches = re.findall(
                        r"^.*?(?=python)", response, re.DOTALL
                    )
                elif "import" in response:
                    algorithm_matches = re.findall(
                        r"^.*?(?=import)", response, re.DOTALL
                    )
                else:
                    algorithm_matches = re.findall(r"^.*?(?=def)", response, re.DOTALL)

            code_matches = re.findall(r"import.*return", response, re.DOTALL)
            if len(code_matches) == 0:
                code_matches = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        # Fallback to the first matched strings
        algorithm: str = algorithm_matches[0] if algorithm_matches else ""
        code: str = code_matches[0] if code_matches else ""

        # Append function outputs to code snippet (the original logic merges them)
        code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)
        return [code_all, algorithm]

    ############################################################################
    # Exposed Methods (Operators)
    ############################################################################

    def i1(self) -> list[str]:
        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ i1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e1(self, parents: list[dict[str, str]]) -> list[str]:
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ e1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e2(self, parents: list[dict[str, str]]) -> list[str]:
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ e2 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m1(self, parents: dict[str, str]) -> list[str]:
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ m1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m2(self, parents: dict[str, str]) -> list[str]:
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ m2 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m3(self, parents: dict[str, str]) -> list[str]:
        prompt_content = self.get_prompt_m3(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ m3 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def h1(self, parents: list[dict[str, str]]) -> list[str]:
        prompt_content = self.get_prompt_h1(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ h1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def f1(self, parent: dict[str, str], failure_cases: str) -> list[str]:
        prompt_content = self.get_prompt_f1(parent, failure_cases)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ f1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def c1(self, parent: dict[str, str], constraints: str) -> list[str]:
        prompt_content = self.get_prompt_c1(parent, constraints)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ c1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def p1(self, parent: dict[str, str], complexity_target: str) -> list[str]:
        prompt_content = self.get_prompt_p1(parent, complexity_target)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ p1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
