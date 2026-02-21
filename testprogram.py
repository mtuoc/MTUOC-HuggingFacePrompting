import sys
from hf_engine import HFModelEngine

def main():
    # 1. Instanciem el motor de Hugging Face
    # El motor llegirà automàticament 'config_tester.yaml'
    engine = HFModelEngine("config.yaml")
    
    # 2. Carreguem el model
    if not engine.load_model():
        print("Error: No s'ha pogut carregar el model. Revisa la configuració.")
        return

    # 3. Configuració de fitxers (podem agafar-ho del YAML o de la línia de comandes)
    input_file = "totranslate1.txt"
    output_file = "translated_batch.txt"
    
    print(f"Començant el processament de: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                text_to_translate = line.strip()
                if not text_to_translate:
                    continue
                
                # Creem el prompt (aquí podries automatitzar el format)
                # Per exemple, si el teu prompt_template al YAML ja espera P[0]
                # podries adaptar la crida de 'generate' per acceptar variables.
                prompt = f"Translate from Russian to Catalan:\nRussian: {text_to_translate}\nCatalan:"
                
                # 4. Generem la resposta
                # raw_text és la resposta bruta, final_text és la processada (JSON/Regex)
                raw_text, final_text = engine.generate(prompt)
                
                # 5. Desem el resultat
                f_out.write(f"{text_to_translate}\t{final_text}\n")
                print(f"Traduït: {text_to_translate[:30]}... -> {final_text[:30]}...")

        print(f"\nProcessament finalitzat. Resultats a: {output_file}")

    except FileNotFoundError:
        print(f"Error: El fitxer '{input_file}' no existeix.")
    except Exception as e:
        print(f"Error inesperat: {e}")

if __name__ == "__main__":
    main()
