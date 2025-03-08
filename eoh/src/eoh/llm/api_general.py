import http.client
import json

class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        # The api_endpoint should be only the hostname,
        # e.g., "dashscope.aliyuncs.com"
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

        # Choose the proper path based on the model.
        # For Qwen Max models, the API requires the "compatible-mode" prefix.
        if self.model_LLM.lower() in ['qwen-max', 'qwen-max-latest']:
            self.api_path = "/compatible-mode/v1/chat/completions"
        else:
            self.api_path = "/v1/chat/completions"

    def get_response(self, prompt_content):
        # Construct the messages payload.
        # The Qwen API example includes a system message; adjust as needed.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_content}
        ]
        payload = json.dumps({
            "model": self.model_LLM,
            "messages": messages,
        })

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": "1",
        }
        
        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                # Establish an HTTPS connection using only the hostname.
                conn = http.client.HTTPSConnection(self.api_endpoint)
                # Use the api_path set in the constructor.
                conn.request("POST", self.api_path, payload, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if self.debug_mode:
                    print("Error in API. Restarting the process...", str(e))
                continue

        return response
