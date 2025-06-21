import requests
import logging
import re
import time
from PyQt6.QtCore import QThread, pyqtSignal

class TranslationWorker(QThread):
    progress = pyqtSignal(int, str, bool)
    finished = pyqtSignal()
    
    def __init__(
        self,
        paragraphs_to_translate,
        llm_instruction,
        context_size,
        temperature,
        all_paragraphs,
        llm_choice="LM Studio",
        model_name="local-model",
        openrouter_api_key=None,
        custom_ollama_prompt=None,
        custom_system_prompt=None,
        custom_user_prompt=None
    ):
        super().__init__()
        self.paragraphs_to_translate = paragraphs_to_translate
        self.llm_instruction = llm_instruction
        self.context_size = context_size
        self.temperature = temperature
        self.all_paragraphs = all_paragraphs
        self.llm_choice = llm_choice
        self.model_name = model_name
        self.openrouter_api_key = openrouter_api_key
        # New parameters
        self.custom_ollama_prompt = custom_ollama_prompt
        self.custom_system_prompt = custom_system_prompt
        self.custom_user_prompt = custom_user_prompt
    
    def split_prefix_suffix(self, text: str):
        m = re.match(r'^(\s*\d+[\.\)]\s*)(.*?)([\.\?!]?)(\s*)$', text)
        if m:
            prefix, core, punct, trail = m.groups()
            return prefix, core, punct + trail
        return "", text, ""
    
    def call_ollama_api(self, prompt):
        """Call Ollama API with proper endpoint and format"""
        headers = {"Content-Type": "application/json"}
        
        # Base URL for Ollama
        base_url = "http://localhost:11434"
        
        # FIRST check if Ollama server responds
        try:
            test_response = requests.get(f"{base_url}", timeout=5)
            logging.debug(f"Ollama base URL response: {test_response.status_code}")
        except Exception as e:
            raise Exception(f"Ollama server not responding at {base_url}. Run: ollama serve. Error: {e}")
        
        # Check /api/tags
        try:
            health_response = requests.get(f"{base_url}/api/tags", timeout=5)
            logging.debug(f"Ollama /api/tags response: {health_response.status_code}")
            if health_response.status_code != 200:
                raise Exception(f"Ollama /api/tags not responding. Status: {health_response.status_code}. Run: ollama serve")
        except requests.exceptions.ConnectionError:
            raise Exception("Ollama server not running. Run: ollama serve")
        except requests.exceptions.Timeout:
            raise Exception("Connection timeout to Ollama. Run: ollama serve")
        
        # Check if model exists
        try:
            models_data = health_response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            logging.debug(f"Available models: {available_models}")
            if self.model_name not in available_models:
                model_base = self.model_name.split(':')[0]
                matching_models = [m for m in available_models if m.startswith(model_base)]
                if matching_models:
                    logging.warning(f"Using model: {matching_models[0]} instead of {self.model_name}")
                    self.model_name = matching_models[0]
                else:
                    raise Exception(f"Model '{self.model_name}' not available. Available: {available_models}")
        except Exception as e:
            logging.warning(f"Cannot check models: {e}")
        
        # Ollama uses /api/generate endpoint
        ollama_payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            "stop": ["<|im_sep|>", "<|im_end|>", "---", "\n\n---", "Human:", "Assistant:"]
        }
        
        generate_url = f"{base_url}/api/generate"
        logging.debug(f"Calling Ollama: {generate_url} with model: {self.model_name}")
        
        response = requests.post(
            generate_url, 
            headers=headers, 
            json=ollama_payload, 
            timeout=None
        )
        
        logging.debug(f"Ollama response status: {response.status_code}")
        logging.debug(f"Ollama response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            logging.error(f"Ollama error {response.status_code}: {response.text}")
            
        response.raise_for_status()
        data = response.json()
        
        # Ollama returns response in 'response' field
        if 'response' not in data:
            raise Exception(f"Invalid response from Ollama: {data}")
        
        response_text = data.get('response', '').strip()
        
        # Additional response cleaning
        response_text = response_text.replace('<|im_sep|>', '')
        response_text = response_text.replace('<|im_end|>', '')
        response_text = response_text.replace('<|im_start|>', '')
        response_text = response_text.strip('-').strip()
        
        return response_text
    
    def call_lm_studio_api(self, system_prompt, user_prompt):
        """Call LM Studio API with chat completions format"""
        headers = {"Content-Type": "application/json"}
        system_msg = {"role": "system", "content": system_prompt}
        user_msg = {"role": "user", "content": user_prompt}
        payload = {
            "model": "local-model",
            "messages": [system_msg, user_msg],
            "temperature": self.temperature,
            "stop": ["\n\n"]
        }
        logging.debug("LM Studio payload messages:")
        logging.debug(" SYSTEM: %s", system_msg["content"])
        logging.debug("  USER: %s", user_msg["content"])
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=None
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()

    def call_openrouter_api(self, system_prompt, user_prompt):
        """Call Openrouter API with chat completions format"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_api_key}"
        }
        system_msg = {"role": "system", "content": system_prompt}
        user_msg = {"role": "user", "content": user_prompt}
        payload = {
            "model": self.model_name,
            "messages": [system_msg, user_msg],
            "temperature": self.temperature,
            "stop": ["\n\n"]
        }
        logging.debug("Openrouter payload messages:")
        logging.debug(" SYSTEM: %s", system_msg["content"])
        logging.debug("  USER: %s", user_msg["content"])
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=None
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()

    def run(self):
        for idx, original_text in self.paragraphs_to_translate:
            try:
                prefix, core_text, suffix = self.split_prefix_suffix(original_text)
                
                # Prepare context
                start_idx = max(0, idx - self.context_size)
                context_paragraphs = []
                for i in range(start_idx, idx):
                    para = self.all_paragraphs[i]
                    context_paragraphs.append(
                        para['translated_text'] if para['is_translated'] else para['original_text']
                    )
                context = "\n".join(context_paragraphs)
                logging.debug(f"Context for paragraph {idx}: \n{context}")
                
                if self.llm_choice == "Ollama":
                    if self.custom_ollama_prompt:
                        full_prompt = self.custom_ollama_prompt.format(
                            context=context,
                            core_text=core_text
                        )
                    else:
                        full_prompt = (
                            self.llm_instruction.strip() + "\n\n"
                            "Context (ONLY for understanding, DO NOT translate):\n"
                            f"{context}\n---\n"
                            "Translate ONLY this (do not write anything else):\n{core_text}"
                        )
                    translated_core = self.call_ollama_api(full_prompt)

                elif self.llm_choice == "Openrouter":
                    if self.custom_system_prompt and self.custom_user_prompt:
                        system_prompt = self.custom_system_prompt.format(context=context)
                        user_prompt   = self.custom_user_prompt.format(core_text=core_text)
                    else:
                        system_prompt = (
                            self.llm_instruction.strip() + "\n\n"
                            "Context (ONLY for understanding, DO NOT translate):\n"
                            f"{context}\n---"
                        )
                        user_prompt = f"Translate ONLY this:\n{core_text}"
                    
                    # Call Openrouter and then wait 3 seconds to respect rate limit
                    translated_core = self.call_openrouter_api(system_prompt, user_prompt)
                    time.sleep(3)

                else:  # LM Studio
                    if self.custom_system_prompt and self.custom_user_prompt:
                        system_prompt = self.custom_system_prompt.format(context=context)
                        user_prompt   = self.custom_user_prompt.format(core_text=core_text)
                    else:
                        system_prompt = (
                            self.llm_instruction.strip() + "\n\n"
                            "Context (ONLY for understanding, DO NOT translate):\n"
                            f"{context}\n---"
                        )
                        user_prompt = f"Translate ONLY this:\n{core_text}"
                    translated_core = self.call_lm_studio_api(system_prompt, user_prompt)
                
                if not translated_core:
                    raise ValueError("Empty translation received")
                
                full_translation = f"{prefix}{translated_core}{suffix}"
                self.all_paragraphs[idx]['translated_text'] = full_translation
                self.all_paragraphs[idx]['is_translated']    = True
                self.progress.emit(idx, full_translation, False)
            
            except Exception as e:
                error_msg = f"ERROR: {e}"
                logging.error(f"Translation error for idx={idx}: {e}")
                self.all_paragraphs[idx]['translated_text'] = error_msg
                self.all_paragraphs[idx]['is_translated']    = False
                self.progress.emit(idx, error_msg, True)
        
        self.finished.emit()