import torch
import json
import re
import yaml
from transformers import pipeline, AutoTokenizer

class HFModelEngine:
    def __init__(self, config_path="config_tester.yaml"):
        self.config = self.load_config(config_path)
        self.pipe = None
        self.tokenizer = None

    def load_config(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading YAML: {e}")
            return None

    def load_model(self, status_callback=None):
        if not self.config: return False
        m_cfg = self.config['model_settings']
        
        if status_callback: status_callback("LOADING MODEL...")
        
        try:
            self.pipe = pipeline(
                "text-generation",
                model=m_cfg['name'],
                device=0 if torch.cuda.is_available() and m_cfg['device'] == 'cuda' else -1,
                trust_remote_code=m_cfg.get('trust_remote_code', True)
            )
            self.tokenizer = self.pipe.tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if status_callback: status_callback("READY")
            return True
        except Exception as e:
            if status_callback: status_callback(f"ERROR: {str(e)[:20]}")
            return False

    def generate(self, prompt, override_regex=None):
        if not self.pipe: return "Error: Model not loaded."
        
        g_cfg = self.config['generation_params']
        temp = float(g_cfg.get('temperature', 0.0))
        
        gen_args = {
            "max_new_tokens": g_cfg.get('max_new_tokens', 128),
            "repetition_penalty": float(g_cfg.get('repetition_penalty', 1.2)),
            "no_repeat_ngram_size": int(g_cfg.get('no_repeat_ngram_size', 0)),
            "do_sample": temp > 0,
            "generation_config": None,
            "return_full_text": False,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        # Stop sequences
        stops = [s.replace("\\n", "\n") for s in g_cfg.get('stop_sequences', [])]
        if stops:
            gen_args["stop_strings"] = stops
            gen_args["tokenizer"] = self.tokenizer

        if gen_args["do_sample"]:
            gen_args["temperature"] = temp
            gen_args["top_k"] = g_cfg.get('top_k', 40)
            gen_args["top_p"] = g_cfg.get('top_p', 0.9)

        res = self.pipe(prompt, **gen_args)
        raw_text = res[0]['generated_text'].strip()
        
        # Post-process (JSON -> Regex)
        return raw_text, self.post_process(raw_text, override_regex)

    def post_process(self, text, override_regex=None):
        processed = text
        p_cfg = self.config.get('prompt_settings', {})

        # 1. JSON Extraction
        json_key = p_cfg.get('json_key', "None")
        if json_key and json_key != "None":
            try:
                json_match = re.search(r'\{.*\}', processed, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    processed = str(data.get(json_key, processed))
            except: pass

        # 2. Regex Filter
        regex_pattern = override_regex if override_regex else p_cfg.get('regex_pattern', "None")
        if regex_pattern and regex_pattern != "None":
            try:
                match = re.search(regex_pattern, processed, re.MULTILINE | re.DOTALL)
                if match:
                    processed = match.group(1) if match.groups() else match.group(0)
            except: pass
            
        return processed.strip()
