import re
import torch
import string
import pandas as pd
from os import path
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForCausalLM,
    TrainingArguments,
    GenerationConfig)
from peft import  (
    prepare_model_for_kbit_training,
    LoraConfig)
from tqdm import tqdm


def prepare_training_arguments(test:str, output_dir:str, num_train_epochs:int):
    training_arguments=None
    if test:
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            do_eval=True,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=4,
            logging_dir=output_dir,
            log_level="debug",
            optim="paged_adamw_32bit",
            save_steps=2, #change to 500, test: 2
            logging_steps=1, #change to 100, test: 1
            learning_rate=1e-4,
            eval_steps=5, #change to 200, test: 5
            bf16=True,
            max_grad_norm=0.3,
            # num_train_epochs=3, # remove "#"
            max_steps=10, #remove this
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
    )
    else:
        training_arguments = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                do_eval=True,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                per_device_eval_batch_size=4,
                log_level="debug",
                optim="paged_adamw_32bit",
                logging_steps=100, 
                learning_rate=1e-4,
                bf16=True,
                max_grad_norm=0.3,
                num_train_epochs=num_train_epochs, 
                warmup_ratio=0.03,
                lr_scheduler_type="constant"
        )

    return training_arguments

def prepare_lora_arguments(lora_alpha:int, r:int):
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        r=r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj","v_proj"])
    return peft_config



def prep_tokenizer(model_path:str, add_eos_token:bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_eos_token=add_eos_token)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.padding_side = 'left'
    return tokenizer

def prepare_model(model_path, tokenizer, quantize:bool, load_in_4bit:bool, load_in_8bit:bool):
    compute_dtype = "bfloat16"
    model = None
    
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,

        )
    elif load_in_8bit:
        bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=compute_dtype,
                bnb_8bit_use_double_quant=True,
        )

    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb_config, device_map={"": 0}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,  device_map={"": 0}
        )
    
    #Resize the embeddings
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16) # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
    model = prepare_model_for_kbit_training(model) # ?
    return model

def get_all_linear_layers(model):
    model_modules = str(model.modules)
    pattern = r'\((\w+)\): Linear'
    linear_layer_names = re.findall(pattern, model_modules)
    
    names = []
    for name in linear_layer_names:
        names.append(name)
        
    target_modules_all_linear_layers = list(set(names))
    return target_modules_all_linear_layers




# f1 score
def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

def white_space_fix(text):
    return " ".join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)



def evaluate_model_for_f1_score(model, tokenizer, dataset: str, result_csv_path: str, test_row_count: int=-1):
    
    results = {"reference": [],
               "question": [],
               "correct_answer": [],
               "answer_w_ref": []}

    f1_all = 0
    with tqdm(total=len(dataset.iloc[:test_row_count]), desc="Testing model") as pbar:
        for index, row in dataset.iloc[:test_row_count].iterrows():
            given_answer_ = generate_answer(row["context"], row["question"], tokenizer, model)
            given_answer = given_answer_.replace("</s>", '')
            
            f1_score = compute_f1(given_answer, row["answer"])
            f1_all = f1_all + f1_score
            
            results["reference"].append(row["context"])
            results["question"].append(row["question"])
            results["correct_answer"].append(row["answer"])
            results["answer_w_ref"].append(given_answer_)
            
            pbar.update(1)
            
    results_df=pd.DataFrame(results)
    results_df.to_csv(path.join(result_csv_path,"test_results.csv"), sep=';', index=True)

    f1_avg = f1_all / len(dataset)
    return f1_avg




def generate_answer(context: str, question: str, tokenizer, model, use_source: bool = True):
    prompt = None
    if use_source:
        prompt = ("Válaszold meg az alábbi kérdést a forrás alapján! Ha a forrás alapján nem megválaszolható a kérdés, mondd hogy: Nincs elegendő adat a kérdés megválaszolásához."
                  + "\n### Forrás:\n" + str(context)
                  + "\n### Kérdés:\n" + str(question)
                  + "\n### Válasz:\n")
    else:
        prompt = ("Válaszold meg az alábbi kérdést"
                 + "\n### Kérdés:\n" + str(question)
                 + "\n### Válasz:\n")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_config = GenerationConfig(
        do_sample=True,
        # top_p=1.0,
        # top_k=50,
        # num_beams=1,
        temperature=0.2,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=512)

    generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config)

    for seq in generation_output.sequences:
        output = tokenizer.decode(seq)
        return output.split("### Válasz:\n")[1].strip()
