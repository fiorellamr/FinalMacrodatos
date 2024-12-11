import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Ruta del corpus de entrenamiento")
parser.add_argument("--model_output", type=str, required=True, help="Directorio de salida del modelo")
args = parser.parse_args()

# Cargar el corpus
dataset = load_dataset('text', data_files={'train': args.input})

# Configurar el modelo base
model_name = "gpt2"
# Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenizar los datos
# def tokenize_function(examples):
#     return tokenizer(
#         examples["text"],
#         truncation=True,
#         padding="max_length",  # Rellena las secuencias más cortas
#         max_length=512         # Define una longitud fija para las secuencias
#     )

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # Rellenar secuencias al tamaño máximo
        max_length=512         # Define la longitud máxima de las secuencias
    )
    tokens["labels"] = tokens["input_ids"].copy()  # Añadir los labels
    return tokens

# Asignar el eos_token del tokenizer como pad_token
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir=args.model_output,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_dir='./logs'
)

# Configurar el data collator con relleno dinámico
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# Configurar el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

trainer.train()
trainer.save_model(args.model_output)
