import csv
import yaml
import re
import codecs
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


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
    """
    Carrega el model i el tokenitzador de Hugging Face amb suport per a 
    tokens especials i configuració de dispositiu.
    """
    print(f"Loading model '{model_name}' (this might take a while)...")
    try:
        # Carreguem el tokenitzador
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Configurem el pad_token si no existeix (comú en models Llama/Salamandra)
        # Això evita errors de configuració en la generació.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Carreguem el model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            # Utilitzem float16 si hi ha GPU disponible per estalviar memòria i guanyar velocitat
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
    """Genera una resposta utilitzant paràmetres configurables des del YAML."""
    try:
        temp = float(options.get("temperature", 0.0))
        do_sample = temp > 0
        
        # Paràmetres base
        gen_args = {
            "max_new_tokens": int(options.get("max_new_tokens", 128)),
            "repetition_penalty": float(options.get("repeat_penalty", 1.0)),
            "do_sample": do_sample,
            "generation_config": None,
        }

        # Gestió de l'EOS i PAD tokens
        if options.get("use_eos_token", True):
            gen_args["eos_token_id"] = pipe.tokenizer.eos_token_id
            gen_args["pad_token_id"] = pipe.tokenizer.pad_token_id or pipe.tokenizer.eos_token_id

        # Gestió de seqüències de parada (stop_sequences)
        stop_seqs = options.get("stop_sequences", [])
        if stop_seqs:
            # Netegem els strings (per si el YAML posa "\n" com a text literal)
            processed_stop_seqs = [s.replace("\\n", "\n") for s in stop_seqs]
            gen_args["stop_strings"] = processed_stop_seqs
            gen_args["tokenizer"] = pipe.tokenizer

        # Paràmetres de sampling
        if do_sample:
            gen_args["temperature"] = temp
            gen_args["top_p"] = float(options.get("top_p", 0.9))
            gen_args["top_k"] = int(options.get("top_k", 40))

        # Execució
        out = pipe(prompt, return_full_text=False, **gen_args)
        return out[0]['generated_text'].strip()

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
