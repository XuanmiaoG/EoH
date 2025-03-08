import json
import textwrap
from ...llm.interface_LLM import InterfaceLLM  # Uncomment and update if needed


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
        We handle generation of new algorithm code strings via an LLM,
        ensuring the LLM outputs valid JSON containing 'algorithm' and 'code'.

        Strict Thought:
          - We store essential info (prompt_task, etc.) from 'prompts'.
          - We do not rely on 'typing' imports; we use native 3.10 union syntax.
          - We will do post-processing to remove extraneous indentation from code.
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

    ############################################################################
    # Prompt Generators (requesting JSON with keys "algorithm" and "code")
    ############################################################################

    def get_prompt_i1(self) -> str:
        """
        Build the prompt content for operator 'i1' (e.g., initial algorithm design).

        Returns:
            A string containing the prompt instructions for the LLM.
        """
        # We explicitly request valid JSON with 'algorithm' and 'code'
        prompt_content = (
            self.prompt_task + "\n"
            "Please return your answer in **valid JSON** format with the following structure:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            "Important:\n"
            " - 'algorithm' must be a single-sentence description in curly braces\n"
            " - 'code' must be the entire Python code\n"
            " - No extra keys, no commentary outside the JSON.\n\n"
            "Now describing your new algorithm:\n"
            "1. Provide the single-sentence description in curly braces.\n"
            "2. Implement the function named "
            + self.prompt_func_name
            + ", which accepts "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ", returning "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            "No further explanation. Output must be **valid JSON** with keys 'algorithm' and 'code'."
        )
        return prompt_content

    def get_prompt_e1(self, indivs: list[dict[str, str]]) -> str:
        """
        Build the prompt content for operator 'e1'.
        Uses a set of existing algorithms (indivs) to create something totally different.
        """
        prompt_indiv = ""
        for i, indiv in enumerate(indivs):
            prompt_indiv += (
                f"No.{i+1} algorithm + code:\n"
                + indiv["algorithm"]
                + "\n"
                + indiv["code"]
                + "\n"
            )

        prompt_content = (
            self.prompt_task + "\n"
            "Here are some existing algorithms:\n"
            + prompt_indiv
            + "Please create a NEW algorithm that is totally different.\n\n"
            "Return your answer in **valid JSON** format:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            "Implement a function named "
            + self.prompt_func_name
            + " with "
            + str(len(self.prompt_func_inputs))
            + " inputs: "
            + self.joined_inputs
            + ", returning "
            + str(len(self.prompt_func_outputs))
            + " outputs: "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\nNo further explanation. Must be valid JSON with 'algorithm' and 'code'."
        )
        return prompt_content

    def get_prompt_e2(self, indivs: list[dict[str, str]]) -> str:
        """
        Build the prompt content for operator 'e2'.
        Motivates a new algorithm based on the common backbone of existing ones.
        """
        prompt_indiv = ""
        for i, indiv in enumerate(indivs):
            prompt_indiv += (
                f"No.{i+1} algorithm + code:\n"
                + indiv["algorithm"]
                + "\n"
                + indiv["code"]
                + "\n"
            )

        prompt_content = (
            self.prompt_task + "\n"
            "We have these existing algorithms:\n"
            + prompt_indiv
            + "Please create a NEW algorithm that is different yet motivated by them.\n\n"
            "Return your answer in **valid JSON** format:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            "Implement a function named "
            + self.prompt_func_name
            + " with "
            + str(len(self.prompt_func_inputs))
            + " inputs: "
            + self.joined_inputs
            + ", returning "
            + str(len(self.prompt_func_outputs))
            + " outputs: "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\nNo extra explanation. Must be valid JSON with 'algorithm' and 'code'."
        )
        return prompt_content

    def get_prompt_m1(self, indiv1: dict[str, str]) -> str:
        """
        Build the prompt content for operator 'm1'.
        Creates a new algorithm as a modified version of the given one.
        """
        prompt_content = (
            self.prompt_task + "\n"
            "Current algorithm + code:\n"
            + indiv1["algorithm"]
            + "\n"
            + indiv1["code"]
            + "\n\n"
            "Please create a new algorithm in a different form but as a modification of the above.\n\n"
            "Return your answer in **valid JSON** format:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            "Implement function "
            + self.prompt_func_name
            + " with "
            + str(len(self.prompt_func_inputs))
            + " inputs: "
            + self.joined_inputs
            + ", returning "
            + str(len(self.prompt_func_outputs))
            + " outputs: "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\nNo extra explanation. Must be valid JSON with 'algorithm' and 'code'."
        )
        return prompt_content

    # You'd do a similar removal of triple-quote indentation in get_prompt_m2, m3, h1, f1, c1, p1
    # The pattern is the same: no leading spaces, no triple-quoted multiline with indentation.

    def get_prompt_m2(self, indiv1: dict[str, str]) -> str:
        prompt_content = (
            self.prompt_task + "\n"
            "Current algorithm + code:\n"
            + indiv1["algorithm"]
            + "\n"
            + indiv1["code"]
            + "\n\n"
            "Identify the main parameters and create a new algorithm with different parameter settings.\n\n"
            "Return your answer in **valid JSON**:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            "Implement function "
            + self.prompt_func_name
            + " with "
            + str(len(self.prompt_func_inputs))
            + " inputs: "
            + self.joined_inputs
            + ", returning "
            + str(len(self.prompt_func_outputs))
            + " outputs: "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\nNo extra explanation. Must be valid JSON with 'algorithm' and 'code'."
        )
        return prompt_content

    def get_prompt_m3(self, indiv1: dict[str, str]) -> str:
        prompt_content = (
            self.prompt_task + "\n"
            "Below is the code to be simplified:\n" + indiv1["code"] + "\n\n"
            "Simplify it to avoid overfitting, keeping function name, inputs, outputs.\n\n"
            "Return your answer in **valid JSON**:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            + self.prompt_inout_inf
            + "\nNo extra explanation. Must be valid JSON with 'algorithm' and 'code'."
        )
        return prompt_content

    def get_prompt_h1(self, indivs: list[dict[str, str]]) -> str:
        prompt_indiv = ""
        for i, indiv in enumerate(indivs):
            prompt_indiv += (
                f"No.{i+1} algorithm + code:\n"
                + indiv["algorithm"]
                + "\n"
                + indiv["code"]
                + "\n"
            )

        prompt_content = (
            self.prompt_task + "\n"
            "You have multiple algorithms:\n"
            + prompt_indiv
            + "Please create a NEW hybrid algorithm by combining strengths.\n\n"
            "Return your answer in **valid JSON**:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            "Implement function "
            + self.prompt_func_name
            + " with "
            + str(len(self.prompt_func_inputs))
            + " inputs, returning "
            + str(len(self.prompt_func_outputs))
            + " outputs. "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\nNo extra explanation. Must be valid JSON with 'algorithm' and 'code'."
        )
        return prompt_content

    def get_prompt_f1(self, indiv1: dict[str, str], failure_cases: str) -> str:
        prompt_content = (
            self.prompt_task + "\n"
            "We have an algorithm:\n"
            + indiv1["algorithm"]
            + "\n"
            + indiv1["code"]
            + "\n"
            "Failure cases:\n" + failure_cases + "\n\n"
            "Improve the algorithm by addressing these failures.\n\n"
            "Return in **valid JSON**:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            "Implement function "
            + self.prompt_func_name
            + " with "
            + str(len(self.prompt_func_inputs))
            + " inputs, returning "
            + str(len(self.prompt_func_outputs))
            + " outputs. "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\nNo extra explanation. Must be valid JSON."
        )
        return prompt_content

    def get_prompt_c1(self, indiv1: dict[str, str], constraints: str) -> str:
        prompt_content = (
            self.prompt_task + "\n"
            "We have code:\n" + indiv1["algorithm"] + "\n" + indiv1["code"] + "\n"
            "Constraints:\n" + constraints + "\n\n"
            "Optimize the algorithm to handle these constraints.\n\n"
            "Return in **valid JSON**:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            "Implement function "
            + self.prompt_func_name
            + " with "
            + str(len(self.prompt_func_inputs))
            + " inputs, returning "
            + str(len(self.prompt_func_outputs))
            + " outputs. "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\nNo extra explanation. Must be valid JSON."
        )
        return prompt_content

    def get_prompt_p1(self, indiv1: dict[str, str], complexity_target: str) -> str:
        prompt_content = (
            self.prompt_task + "\n"
            "We have code:\n" + indiv1["algorithm"] + "\n" + indiv1["code"] + "\n"
            "Complexity target:\n" + complexity_target + "\n\n"
            "Add complexity layers while preserving interpretability.\n\n"
            "Return in **valid JSON**:\n"
            "{\n"
            '  "algorithm": "...",\n'
            '  "code": "..."\n'
            "}\n"
            "Implement function "
            + self.prompt_func_name
            + " with "
            + str(len(self.prompt_func_inputs))
            + " inputs, returning "
            + str(len(self.prompt_func_outputs))
            + " outputs. "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\nNo extra explanation. Must be valid JSON."
        )
        return prompt_content

    ############################################################################
    # Core JSON Parsing with Post-Processing
    ############################################################################

    def _postprocess_code(self, code: str) -> str:
        """
        Strict Thought:
          1. We attempt to 'dedent' the code to remove uniform leading indentation blocks.
          2. Then we can remove leading/trailing blank lines.
          3. Optionally we remove tabs or weird spacing with a re.sub.
          The ultimate goal is to reduce chance of 'unexpected indent' errors.

        Args:
            code: The raw code string from LLM JSON.
        Returns:
            A post-processed code string, hopefully free of random leading indentation.
        """
        # Step 1: dedent (handles uniform left indentation)
        dedented = textwrap.dedent(code)

        # Step 2: strip leading/trailing blank lines
        dedented = dedented.strip("\n")

        # Step 3: optionally remove extraneous tabs or repeated spaces
        # e.g. replace tabs with 4 spaces
        dedented = dedented.replace("\t", "    ")

        # Another optional fix: remove repeated blank lines
        # dedented = re.sub(r"\n\s*\n\s*\n+", "\n\n", dedented)

        return dedented

    def _get_alg(self, prompt_content: str) -> list[str]:
        """
        Query the LLM, parse JSON.
        Then post-process 'code' to remove indentation issues.

        Example math formula (strict):
            Let R be the LLM response => O = json.loads(R)
            => time ~ O(len(R))

        Args:
            prompt_content: The prompt string to send.
        Returns:
            [code_all, algorithm]
        """
        max_retries = 3
        n_retry = 1
        algorithm = ""
        code = ""

        while n_retry <= max_retries:
            response = self.interface_llm.get_response(prompt_content)
            try:
                data = json.loads(response)
                if "algorithm" not in data or "code" not in data:
                    raise ValueError("Missing 'algorithm' or 'code' in JSON.")

                # We do post-processing to code
                raw_code = data["code"]
                cleaned_code = self._postprocess_code(raw_code)

                # store results
                algorithm = data["algorithm"]
                code = cleaned_code
                break
            except (json.JSONDecodeError, ValueError) as e:
                if self.debug_mode:
                    print(f"[Attempt {n_retry}] => invalid JSON or missing keys: {e}")
                prompt_content = (
                    "Your last output was not valid JSON with the required keys. "
                    "Please return valid JSON with keys 'algorithm' and 'code' only."
                )
            n_retry += 1

        # Optionally, add function outputs to code snippet
        code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)
        return [code_all, algorithm]

    ############################################################################
    # Exposed Methods (Operators)
    ############################################################################

    def i1(self) -> list[str]:
        """
        Operator i1: no parents. Return code + alg.
        """
        prompt_content = self.get_prompt_i1()
        if self.debug_mode:
            print(">>> Prompt [i1]:\n", prompt_content)
            input("Press Enter to continue...")

        code_all, algorithm = self._get_alg(prompt_content)
        if self.debug_mode:
            print(">>> 'algorithm':\n", algorithm)
            print(">>> 'code':\n", code_all)
            input("Press Enter to continue...")

        return [code_all, algorithm]

    # Similarly define e1, e2, m1, m2, m3, h1, f1, c1, p1
    # all rely on their respective get_prompt_* methods.

    def e1(self, parents: list[dict[str, str]]) -> list[str]:
        # For brevity, just show the call
        prompt_content = self.get_prompt_e1(parents)
        code_all, algorithm = self._get_alg(prompt_content)
        return [code_all, algorithm]

    def e2(self, parents: list[dict[str, str]]) -> list[str]:
        prompt_content = self.get_prompt_e2(parents)
        code_all, algorithm = self._get_alg(prompt_content)
        return [code_all, algorithm]

    def m1(self, parents: dict[str, str]) -> list[str]:
        prompt_content = self.get_prompt_m1(parents)
        code_all, algorithm = self._get_alg(prompt_content)
        return [code_all, algorithm]

    def m2(self, parents: dict[str, str]) -> list[str]:
        prompt_content = self.get_prompt_m2(parents)
        code_all, algorithm = self._get_alg(prompt_content)
        return [code_all, algorithm]

    def m3(self, parents: dict[str, str]) -> list[str]:
        prompt_content = self.get_prompt_m3(parents)
        code_all, algorithm = self._get_alg(prompt_content)
        return [code_all, algorithm]

    def h1(self, parents: list[dict[str, str]]) -> list[str]:
        prompt_content = self.get_prompt_h1(parents)
        code_all, algorithm = self._get_alg(prompt_content)
        return [code_all, algorithm]

    def f1(self, parent: dict[str, str], failure_cases: str) -> list[str]:
        prompt_content = self.get_prompt_f1(parent, failure_cases)
        code_all, algorithm = self._get_alg(prompt_content)
        return [code_all, algorithm]

    def c1(self, parent: dict[str, str], constraints: str) -> list[str]:
        prompt_content = self.get_prompt_c1(parent, constraints)
        code_all, algorithm = self._get_alg(prompt_content)
        return [code_all, algorithm]

    def p1(self, parent: dict[str, str], complexity_target: str) -> list[str]:
        prompt_content = self.get_prompt_p1(parent, complexity_target)
        code_all, algorithm = self._get_alg(prompt_content)
        return [code_all, algorithm]
