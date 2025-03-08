import re
from ...llm.interface_LLM import InterfaceLLM  # Uncomment or adapt if needed


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
        """
        We generate new algorithm code strings via an LLM,
        instructing it to output the code in a Markdown code block:

            ```python
            def my_func(...):
                ...
            ```

        Then parse it with a regex for triple backticks.
        If multiple attempts fail, we present more forceful instructions,
        and we do minor 'auto-correction' on the code string.

        Strict Thought:
          - Helps reduce partial errors from LLM's output format variations.
          - We do not import from typing, only python3.10 union syntax.
          - We parse single-sentence 'algorithm' from { ... } with a separate regex.
        """
        self.prompt_task: str = prompts.get_task()
        self.prompt_func_name: str = prompts.get_func_name()
        self.prompt_func_inputs: list[str] = prompts.get_func_inputs()
        self.prompt_func_outputs: list[str] = prompts.get_func_outputs()
        self.prompt_inout_inf: str = prompts.get_inout_inf()
        self.prompt_other_inf: str = prompts.get_other_inf()

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

    def _auto_correct_code(self, raw_code: str) -> str:
        """
        Perform a minimal auto-correction on the code:
         - remove carriage returns (\r)
         - remove repeated blank lines

        Strict Thought:
          - Real usage might do deeper transformations or syntax checks (ast.parse).
          - This is a simple demonstration.

        Args:
            raw_code: The code string from LLM.

        Returns:
            A corrected code string.
        """
        # remove windows-style carriage returns
        corrected = raw_code.replace("\r", "")
        # remove repeated blank lines
        corrected = re.sub(r"\n\s*\n\s*\n+", "\n\n", corrected)
        return corrected

    def _get_alg(self, prompt_content: str) -> list[str]:
        """
        Query LLM with 'prompt_content'; parse 'algorithm' from { ... } and parse code
        from triple backtick python code block.

        If we fail to match either part, we do up to 3 attempts. After the second fail,
        we give a more forceful prompt. We also do minimal auto-correction of the code.

        Example math formula (strict):
          - If response length is n, regex cost ~ O(n) each time.

        Args:
            prompt_content: The prompt string we send to the LLM.

        Returns:
            [code_all, algorithm]:
              code_all = (cleaned code block) + appended function outputs,
              algorithm = single-sentence from braces.
        """
        # Patterns for braces and code block
        brace_pattern = re.compile(r"\{([^}]*)\}")
        md_code_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)

        max_retries = 3
        attempt = 1
        algorithm = ""
        code_block = ""

        while attempt <= max_retries:
            response = self.interface_llm.get_response(prompt_content)

            alg_matches = brace_pattern.findall(response)
            code_matches = md_code_pattern.findall(response)

            if alg_matches and code_matches:
                # pick first match from each
                algorithm = alg_matches[0].strip()
                code_block = code_matches[0].strip()
                # do minimal auto-correction
                code_block = self._auto_correct_code(code_block)
                break
            else:
                if self.debug_mode:
                    print(f"[Attempt {attempt}] => Missing braces or code block.")
                # after second attempt, we update the prompt to be more strict
                if attempt == 2:
                    prompt_content = (
                        "You must strictly output:\n"
                        "1) {single-sentence} describing the algorithm\n"
                        "2) ```python code block```\n"
                        "No extra text or commentary, just these two parts!"
                    )
                else:
                    prompt_content = (
                        "Your last output is invalid. Provide {algorithm} and ```python code```. "
                        "No additional commentary!"
                    )
            attempt += 1

        # final code
        code_all = code_block + " " + ", ".join(s for s in self.prompt_func_outputs)
        return [code_all, algorithm]

    def get_prompt_i1(self) -> str:
        """
        Build the prompt content for operator 'i1' (e.g., initial algorithm design).

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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
        """
        Build the prompt content for operator 'e1'.
        This uses a set of existing algorithms (indivs) to create something totally different.

        Args:
            indivs: List of dicts with 'algorithm' and 'code'.

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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
        """
        Build the prompt content for operator 'e2'.
        Motivates a new algorithm based on the common backbone idea of existing ones.

        Args:
            indivs: List of dicts with 'algorithm' and 'code'.

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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
        """
        Build the prompt content for operator 'm1'.
        Creates a new algorithm as a modified version of the given one.

        Args:
            indiv1: A dict with 'algorithm' and 'code'.

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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
        """
        Build the prompt content for operator 'm2'.
        Creates a new algorithm that adjusts parameter settings of the given one.

        Args:
            indiv1: A dict with 'algorithm' and 'code'.

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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
        """
        Build the prompt content for operator 'm3'.
        Simplifies components of the given code to enhance generalization.

        Args:
            indiv1: A dict with 'algorithm' and 'code'.

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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
        """
        Build the prompt content for operator 'h1'.
        Creates a new hybrid algorithm by integrating multiple algorithms.

        Args:
            indivs: List of dicts with 'algorithm' and 'code'.

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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
        """
        Build the prompt content for operator 'f1'.
        Improves the algorithm by analyzing and addressing its failure cases.

        Args:
            indiv1: A dict with 'algorithm' and 'code'.
            failure_cases: A string describing known failure scenarios.

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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
        """
        Build the prompt content for operator 'c1'.
        Creates a constraint-focused algorithm by addressing specified constraints.

        Args:
            indiv1: A dict with 'algorithm' and 'code'.
            constraints: A string describing the critical constraints to handle.

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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
        """
        Build the prompt content for operator 'p1'.
        Increases complexity layers of the given algorithm.

        Args:
            indiv1: A dict with 'algorithm' and 'code'.
            complexity_target: Description of how to add complexity.

        Returns:
            A string containing the prompt instructions for the LLM.
        """
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

    def i1(self) -> list[str]:
        """
        Operator 'i1': no parents, purely new code.
        """
        prompt_content = self.get_prompt_i1()
        if self.debug_mode:
            print("\n>>> Prompt [i1]:\n", prompt_content)
            input("Press Enter to continue...")

        code_all, algorithm = self._get_alg(prompt_content)

        if self.debug_mode:
            print("Algorithm:\n", algorithm)
            print("Code:\n", code_all)
            input("Press Enter to continue...")

        return [code_all, algorithm]

    def e1(self, parents: list[dict[str, str]]) -> list[str]:
        prompt_content = self.get_prompt_e1(parents)
        code_all, alg = self._get_alg(prompt_content)
        return [code_all, alg]

    def e2(self, parents: list[dict[str, str]]) -> list[str]:
        prompt_content = self.get_prompt_e2(parents)
        code_all, alg = self._get_alg(prompt_content)
        return [code_all, alg]

    def m1(self, parents: dict[str, str]) -> list[str]:
        prompt_content = self.get_prompt_m1(parents)
        code_all, alg = self._get_alg(prompt_content)
        return [code_all, alg]

    def m2(self, parents: dict[str, str]) -> list[str]:
        prompt_content = self.get_prompt_m2(parents)
        code_all, alg = self._get_alg(prompt_content)
        return [code_all, alg]

    def m3(self, parents: dict[str, str]) -> list[str]:
        prompt_content = self.get_prompt_m3(parents)
        code_all, alg = self._get_alg(prompt_content)
        return [code_all, alg]

    def h1(self, parents: list[dict[str, str]]) -> list[str]:
        prompt_content = self.get_prompt_h1(parents)
        code_all, alg = self._get_alg(prompt_content)
        return [code_all, alg]

    def f1(self, parent: dict[str, str], failure_cases: str) -> list[str]:
        prompt_content = self.get_prompt_f1(parent, failure_cases)
        code_all, alg = self._get_alg(prompt_content)
        return [code_all, alg]

    def c1(self, parent: dict[str, str], constraints: str) -> list[str]:
        prompt_content = self.get_prompt_c1(parent, constraints)
        code_all, alg = self._get_alg(prompt_content)
        return [code_all, alg]

    def p1(self, parent: dict[str, str], complexity_target: str) -> list[str]:
        prompt_content = self.get_prompt_p1(parent, complexity_target)
        code_all, alg = self._get_alg(prompt_content)
        return [code_all, alg]
