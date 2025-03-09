from ..llm.api_general import InterfaceAPI
from ..llm.api_local_llm import InterfaceLocalLLM


class InterfaceLLM:
    """
    The InterfaceLLM class provides a unified interface to either:
      - A local LLM (via InterfaceLocalLLM), or
      - A remote LLM API (via InterfaceAPI).

    Strict Thought:
      - We do not import 'typing'; we use Python 3.10+ union syntax only.
      - We add docstrings and inline comments to clarify.

    Usage:
      1) Based on the constructor arguments (llm_use_local, api_endpoint, etc.),
         we choose whether to instantiate InterfaceLocalLLM or InterfaceAPI.
      2) We test the chosen LLM by sending a trivial prompt "1+1=?"
         to ensure it responds correctly.
      3) If no valid response, we terminate execution (exit).
    """

    def __init__(
        self,
        api_endpoint: str | None,
        api_key: str | None,
        model_LLM: str | None,
        llm_use_local: bool,
        llm_local_url: str | None,
        debug_mode: bool,
    ) -> None:
        """
        Constructor for InterfaceLLM.

        Args:
            api_endpoint: The remote LLM API endpoint (e.g., 'https://api.example.com/chat'),
                          or None if local LLM is used.
            api_key: The authorization key/token for the remote LLM, or None if local.
            model_LLM: The model name/identifier (for remote usage).
            llm_use_local: Whether we use a local LLM deployment (True) or a remote LLM API (False).
            llm_local_url: The local server URL if llm_use_local=True.
            debug_mode: If True, print debug info.

        Raises:
            SystemExit: If invalid config is found for local or remote usage.
        """
        self.api_endpoint: str | None = api_endpoint
        self.api_key: str | None = api_key
        self.model_LLM: str | None = model_LLM
        self.debug_mode: bool = debug_mode
        self.llm_use_local: bool = llm_use_local
        self.llm_local_url: str | None = llm_local_url

        print("- check LLM API")

        # Decide which LLM interface to use
        if self.llm_use_local:
            # We are using a local LLM
            print("local llm delopyment is used ...")

            # Check if URL is valid
            if self.llm_local_url is None or self.llm_local_url == "xxx":
                print(">> Stop with empty url for local llm !")
                exit()

            # Instantiate the local LLM interface
            self.interface_llm = InterfaceLocalLLM(self.llm_local_url)
        else:
            # We are using a remote LLM API
            print("remote llm api is used ...")

            # Check if API key/endpoint are valid
            if (
                self.api_key is None
                or self.api_endpoint is None
                or self.api_key == "xxx"
                or self.api_endpoint == "xxx"
            ):
                print(
                    ">> Stop with wrong API setting: set api_endpoint (e.g., api.chat...) and api_key (e.g., kx-...) !"
                )
                exit()

            # Instantiate the remote LLM interface
            self.interface_llm = InterfaceAPI(
                self.api_endpoint,
                self.api_key,
                self.model_LLM,
                self.debug_mode,
            )

        # Test call to verify the LLM is responsive
        test_response: str | None = self.interface_llm.get_response("1+1=?")
        if test_response is None:
            print(
                ">> Error in LLM API, wrong endpoint, key, model or local deployment!"
            )
            exit()

    def get_response(self, prompt_content: str) -> str | None:
        """
        Get response from the chosen LLM (local or remote).

        Args:
            prompt_content: The prompt string we send to the LLM.

        Returns:
            The LLM's text response, or None if something went wrong.
        """
        response: str | None = self.interface_llm.get_response(prompt_content)
        return response
