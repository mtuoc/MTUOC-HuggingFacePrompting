import csv
import yaml
import re
import codecs
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Variable de configuració global
CONFIG = {}

def load_config(config_path):
    global CONFIG
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
        return True
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_path}' not found.")
        return False
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML file: {e}")
        return False

def load_hf_model(model_name, device_map="auto"):
    """Carrega el model i el tokenitzador de Hugging Face."""
    print(f"Loading model '{model_name}' (this might take a while)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        # Configurem el pipeline de generació
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        print("Model loaded successfully.")
        return hf_pipeline
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

def obtenir_resposta_hf(prompt: str, pipe, options: dict):
    """Genera una resposta utilitzant el pipeline de Hugging Face."""
    try:
        # Mapatge de paràmetres de YAML a Transformers
        # Si temp és 0, en HF normalment es fa do_sample=False (greedy decoding)
        temp = float(options.get("temperature", 0.0))
        do_sample = temp > 0
        
        gen_args = {
            "max_new_tokens": options.get("max_new_tokens", 128),
            "temperature": temp if do_sample else None,
            "top_p": options.get("top_p", 0.9) if do_sample else None,
            "top_k": options.get("top_k", 40) if do_sample else None,
            "do_sample": do_sample,
            "repetition_penalty": float(options.get("repeat_penalty", 1.0)),
            "pad_token_id": pipe.tokenizer.eos_token_id
        }
        
        # Eliminar claus Nones per evitar errors en el pipeline
        gen_args = {k: v for k, v in gen_args.items() if v is not None}

        out = pipe(prompt, **gen_args)
        # Extraiem només el text generat nou (sense el prompt)
        full_text = out[0]['generated_text']
        new_text = full_text[len(prompt):].strip()
        return new_text
    except Exception as e:
        return f"ERROR: {e}"

def process_file(file_cfg, hf_cfg, prompt_cfg):
    nom_fitxer = file_cfg["input_filename"]
    nom_fitxer_sortida = file_cfg["output_filename"]
    separador = file_cfg["delimiter"]
    
    # 1. Càrrega del Model
    pipe = load_hf_model(hf_cfg["model"])
    if not pipe:
        return

    # 2. Processament de l'Arxiu
    try:
        separator = '\t' if separador == '\\t' else separador
        prompt_template = prompt_cfg["prompt_template"]
        regex_pattern = prompt_cfg["regex_pattern"]
        if regex_pattern == "None": regex_pattern = None

        print(f"\nStart processing '{nom_fitxer}'...")
        
        with open(nom_fitxer, 'r', encoding='utf-8') as input_file, \
            codecs.open(nom_fitxer_sortida, "w", encoding="utf-8") as output_file:
            
            lector = csv.reader(input_file, delimiter=separator)

            for i, fila in enumerate(lector):
                prompt = prompt_template.format(P=fila)
                
                # Cridem a Hugging Face
                response = obtenir_resposta_hf(prompt, pipe, hf_cfg)
                
                if regex_pattern:
                    match = re.search(regex_pattern, response)
                    respostafinal = match.group(1).strip() if match else response
                else:
                    respostafinal = response
                
                print(f"{separator.join(fila)}{separator}{respostafinal}")
                output_file.write(f"{separator.join(fila)}{separator}{respostafinal}\n")
                
        print(f"\nProcess ended. Results saved at: {nom_fitxer_sortida}")

    except Exception as e:
        print(f"\nUnexpected ERROR: {e}")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    if load_config(config_file):
        process_file(
            CONFIG["file_settings"],
            CONFIG["hf_settings"], # Canviat de ollama a hf
            CONFIG["prompt_settings"]
        )
