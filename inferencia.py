import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="Texto de entrada para el modelo")
parser.add_argument("--model_path", type=str, required=True, help="Ruta del modelo entrenado")
args = parser.parse_args()

# Cargar el modelo entrenado
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path)

# Generar texto
inputs = tokenizer(args.prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Salida del modelo:")
print(result)
