import csv
import yaml
import codecs
import sys
from hf_engine import HFModelEngine  # Importem el motor centralitzat

def process_file(config_path):
    # 1. Instanciem el motor amb el fitxer de configuració
    engine = HFModelEngine(config_path)
    if not engine.config:
        return

    file_cfg = engine.config["file_settings"]
    prompt_cfg = engine.config["prompt_settings"]
    
    nom_fitxer = file_cfg["input_filename"]
    nom_fitxer_sortida = file_cfg["output_filename"]
    separador = '\t' if file_cfg["delimiter"] == '\\t' else file_cfg["delimiter"]

    # 2. Carreguem el model a través del motor
    if not engine.load_model(status_callback=print):
        return

    # 3. Processament de l'Arxiu
    try:
        prompt_template = prompt_cfg["prompt_template"]
        
        print(f"\nIniciant el processament de '{nom_fitxer}'...")
        
        with open(nom_fitxer, 'r', encoding='utf-8') as input_file, \
             codecs.open(nom_fitxer_sortida, "w", encoding="utf-8") as output_file:
            
            lector = csv.reader(input_file, delimiter=separador)

            for fila in lector:
                if not fila: continue
                
                # Preparem el prompt amb les dades de la fila
                prompt = prompt_template.format(P=fila)
                
                # Cridem al motor: ell s'encarrega de la generació, 
                # de les stop_sequences, del JSON i del Regex automàticament.
                raw_response, final_response = engine.generate(prompt)
                
                # Escrivim el resultat (Fila original + Traducció filtrada)
                result_line = f"{separador.join(fila)}{separador}{final_response}"
                print(result_line)
                output_file.write(result_line + "\n")
                
        print(f"\nProcés finalitzat. Resultats desats a: {nom_fitxer_sortida}")

    except Exception as e:
        print(f"\nERROR inesperat: {e}")

if __name__ == "__main__":
    # Permet passar el fitxer yaml per argument, si no, usa example1.yaml
    config_file = sys.argv[1] if len(sys.argv) > 1 else "example1.yaml"
    process_file(config_file)
