MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEST_PERCENTAGE = 0.3
NUM_CHOICES = 5
DATASET_DIR = "./dataset_llama"
FINETUNED_MODEL_DIR = "./outputs_llama"
SEED = 595
from transformers import BitsAndBytesConfig, LlamaForCausalLM, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import torch
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import random
import argparse
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import login
from PIL import Image
import zipfile
import wandb
import json


print(f"Loading config.json")
with open('config.json', 'r') as file:
    config = json.load(file)

wandb_api_key = config['wandb_api_key']
hf_token = config['huggingface_hub_token']
print(f"Login wandb...")

wandb.login(key=wandb_api_key)

# Initialize a W&B run
run = wandb.init(
    project='Final Project',
    job_type="training",
    anonymous="allow"
)
print(f"Completed config setup")


def formatting_prompts_func(example):
    """
    Prepares a batch of examples by formatting each into a prompt string suitable for input to a model.
    
    Params:
        example (dict): A single example containing 'input' and 'label'.
    
    Returns:
        list: A list of formatted prompt strings.
    """
    inputs = []

    for i in range(len(example['effects'])):
        inputs.append(create_prompt_instruction(example['effects'][i], example['verb_noun_choices'][i], example['correct_indices'][i]))
    return inputs

def create_prompt_instruction(effect, verb_noun_choices, correct_label=None):
    """
    Creates a formatted instruction prompt for the action-effect dataset.
    
    Params:
        effect (str): The effect sentence.
        correct_label (str): The correct verb-noun pair.
    
    Returns:
        str: A formatted prompt string.
    """
    correct_label = f"{correct_label + 1}" if correct_label is not None else ""
    num_choices = len(verb_noun_choices)
    verb_noun_choices = "\n".join([f"{i+1}: {choice}" for i, choice in enumerate(verb_noun_choices)])
    # instruction = f"""You are given an action-effect question. The question includes some effects and a list of action choices. Based on the provided effects, choose the most possible action choice that causes the effects.\n
    # In your answer, first predict the possible actions based on the effects, then choose the correct action from the list of action choices with a prefix of \'### Correct Label: \' at the end of your answer.\n
    # Below is the question:"""
    # if correct_label is not None:
    #     return f"{instruction}\n### Effect:\n{effect}\n### Action Choices:\n{verb_noun_choices}\n### Correct Label: {correct_label}"
    # else:
    #     return f"{instruction}\n### Effect:\n{effect}\n### Action Choices:\n{verb_noun_choices}"
    
    # prompt = f"""In your answer, first predict the possible actions based on the effects, then choose the correct action (one number from 1 to { str(num_choices) }) from the list of action choices with a prefix of \'### Correct Label: \' at the end of your answer.
    # """
    prompt = f"Select the correct action that would cause the effect depicted in the image."
    instruction = f"""USER: You are a helpful assistant. You are shown an image.\n{prompt}\nAnswer with one number from 1 to { str(num_choices) }.\nEffect:\n{effect}\nChoices:\n{verb_noun_choices}\nASSISTANT: """
    if correct_label is not None:
        return f"{instruction}{correct_label}"
    else:
        return f"{instruction}"


def process_zip(zip_path):
    # Temporary directory for extraction
    extract_dir = "./temp"
    
    # Extract ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Arrays to hold effect phrases and action verbs
    effect_phrases = []
    action_verbs = []
    
    # Walk through extracted directories
    for root, _, files in os.walk(extract_dir):
        action_verb = os.path.basename(root)  # Folder name is the action verb
        for file in files:
            if file.endswith(('.txt', '.doc', '.docx', '.pdf')):  # Check for text files
                file_path = os.path.join(root, file)
                try:
                    # Load effect phrase and append to the list
                    with open(file_path, 'r') as f:
                        effect_phrase = f.read()
                    effect_phrases.append(effect_phrase)
                    action_verbs.append(action_verb)
                except Exception as e:
                    print(f"Error loading effect phrase {file_path}: {e}")
    # Convert action verbs list to a NumPy array
    action_verbs_array = np.array(action_verbs, dtype=str)
    return effect_phrases, action_verbs_array

def get_dataset(params):
    if not os.path.exists(DATASET_DIR) or params.force_reload_dataset:
        dataset = load_dataset("sled-umich/Action-Effect", trust_remote_code=True)
        dataset_effects = [eff for eff_list in dataset["ActionEffect"]["effect_sentence_list"] for eff in eff_list]
        verb_nouns = np.array(dataset["ActionEffect"]["verb noun"], dtype=str)
        dataset_correct_verb_nouns = np.repeat(verb_nouns, [len(eff_list) for eff_list in dataset["ActionEffect"]["effect_sentence_list"]])

        effect_array = dataset_effects
        correct_verb_nouns = dataset_correct_verb_nouns
            
        verb_noun_choices = []
        correct_indices = []
        for i in range(len(effect_array)):
            choices = [correct_verb_nouns[i]]
            rem_indices = set([j for j in range(len(verb_nouns))])
            index = np.where(verb_nouns == correct_verb_nouns[i])[0][0]
            rem_indices.remove(index)
            added_indices = set([index])
            
            while (len(choices) < NUM_CHOICES):
                rand_ind = random.choice(list(rem_indices))
                choices.append(correct_verb_nouns[rand_ind])
                assert(rand_ind not in added_indices)
                rem_indices.remove(rand_ind)
                added_indices.add(rand_ind)

            random.shuffle(choices)
            correct_indices.append(choices.index(correct_verb_nouns[i]))
            verb_noun_choices.append(choices)

        verb_noun_choices = np.array(verb_noun_choices)
        correct_indices = np.array(correct_indices)

        train_effect, test_effect, train_correct_verb_nouns, test_correct_verb_nouns, \
        train_verb_noun_choices, test_verb_noun_choices, train_correct_indices, test_correct_indices = train_test_split(
            effect_array, correct_verb_nouns, verb_noun_choices, correct_indices, test_size=TEST_PERCENTAGE, random_state=42
        )

        train_dict = {
            "effects": train_effect,
            "correct_verb_nouns": train_correct_verb_nouns,
            "verb_noun_choices": train_verb_noun_choices,
            "correct_indices": train_correct_indices,
        }
        test_dict = {
            "effects": test_effect,
            "correct_verb_nouns": test_correct_verb_nouns,
            "verb_noun_choices": test_verb_noun_choices,
            "correct_indices": test_correct_indices,
        }

        # Convert to datasets.Dataset objects
        train_dataset = Dataset.from_dict(train_dict)
        test_dataset = Dataset.from_dict(test_dict)

        # Combine into a DatasetDict if needed
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
        })
        dataset_dict.save_to_disk(DATASET_DIR)
    else:
        print("Dataset already exists. Skipping preprocessing.")
        dataset_dict = DatasetDict.load_from_disk(DATASET_DIR)
    
    return dataset_dict

def save_model(trainer, model, output_dir):
    # Handle saving model in case of distributed/parallel training
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    model_to_save.save_pretrained(output_dir)

    # # Load the LoRA configuration from the saved model directory
    lora_config = LoraConfig.from_pretrained(output_dir)

    # # Reapply the LoRA configuration to the model
    model = get_peft_model(model, lora_config)

    return model

def load_llama(lora_config):
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        # Check for LoRA parameters (they may have names like 'lora_A', 'lora_B', etc.)
        if 'lora' in name.lower():
            param.requires_grad = True  # Enable gradients for LoRA parameters

    model.config.use_cache = False

    return model, tokenizer


class LLamaDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples, split="train", response_template="#### Correct Label: ", mlm=False):
        if split == "train":
            input_texts = [create_prompt_instruction(example['effects'], example['verb_noun_choices'], example['correct_indices']) for example in examples]
        else:
            input_texts = [create_prompt_instruction(example['effects'], example['verb_noun_choices'], None) for example in examples]
        
        # Tokenize the input texts
        tokenized_inputs = self.tokenizer(input_texts, 
                                           padding=True, 
                                           truncation=True, 
                                           return_tensors="pt")  # Return PyTorch tensors
        

        batch = {
            "input_ids": tokenized_inputs["input_ids"],  # Tensor of input token IDs
            "attention_mask": tokenized_inputs["attention_mask"],  # Tensor for attention mask
            "labels": tokenized_inputs["input_ids"].clone()  # Assuming labels are the same as input_ids for training
        }
        
        return batch
        

def train_model(
    dataset_dict,
    r=8,
    lora_alpha=8,
    lora_dropout=0.2,
    target_modules="all-linear",
    num_epochs=3,
    lr=1.5e-5,
    per_device_train_batch_size=8,
    gradient_acc_steps=4,
    logging_steps=10,
    max_grad_norm=1.0,
    max_seq_length=256
    ):
    
    print("r:", r)
    print("lora_alpha:", lora_alpha)
    print("lora_dropout:", lora_dropout)
    print("target_modules:", target_modules)
    print("num_epochs:", num_epochs)
    print("lr:", lr)
    print("per_device_train_batch_size:", per_device_train_batch_size)
    print("gradient_acc_steps:", gradient_acc_steps)
    print("logging_steps:", logging_steps)
    print("max_grad_norm:", max_grad_norm)
    print("max_seq_length:", max_seq_length)
   
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
 
    model, tokenizer = load_llama(lora_config)

    data_collator = LLamaDataCollator(tokenizer)

    train_dataset = dataset_dict["train"]
    print("train shape:", train_dataset.shape)
    # eval_dataset = dataset_dict["test"]
    eval_dataset = train_dataset
    
    
    for i in range(3):
        example = create_prompt_instruction(
                train_dataset[i]['effects'],
                train_dataset[i]['verb_noun_choices'],
                train_dataset[i]['correct_indices'],
            )
        print(example)
        
    training_arguments = SFTConfig(
        run_name="finetune-llama",
        output_dir="./results",
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_acc_steps,
        logging_steps=logging_steps,
        max_grad_norm=max_grad_norm,
        max_seq_length=max_seq_length,
        optim="paged_adamw_32bit",
        lr_scheduler_type="linear",
        do_eval=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb",
        seed=SEED,
    
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        # dataset_text_field="text",  # need a dummy field
        data_collator=data_collator,
        tokenizer=tokenizer,
        # dataset_kwargs={"skip_prepare_dataset": True},
        args=training_arguments,
    )
    
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
    
    if params.zero_shot:
        print("STARTING TRAIN")
        trainer.train()

    return trainer, model, tokenizer

def test(model, dataset, create_prompt_func, tokenizer, max_new_tokens=1):

    total = 0
    num_correct = 0
    print("test size:", dataset.shape)
    for example in dataset:
        messages = example
        text = create_prompt_func(messages["effects"], messages["verb_noun_choices"], None)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        predicted_answer = tokenizer.decode(output[0], skip_special_tokens=True)
        # response = predicted_answer.find("### Correct Label: ") + len("### Correct Label: ")
        response = predicted_answer.find("ASSISTANT: ") + len("ASSISTANT: ")
        response = predicted_answer[response:].strip()
        
        print("Ground Truth:", messages['verb_noun_choices'][messages['correct_indices']], messages["correct_indices"] + 1)
        print("Predicted:", predicted_answer)
        if response == str(messages["correct_indices"] + 1):
            num_correct += 1
        total += 1
        
    print(f"Accuracy: {num_correct/total}")

def load_llama_trained_lora(output_dir):
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig.from_pretrained(output_dir)
    model = get_peft_model(model, lora_config)
    return model

def main(params):
    
    dataset_dict = get_dataset(params)
    if (not params.test_only):
        trainer, model, tokenizer = train_model(dataset_dict)
        model = save_model(trainer, model, FINETUNED_MODEL_DIR)
        test(model, dataset_dict["test"], create_prompt_instruction, tokenizer)

    else:
        model = load_llama_trained_lora(FINETUNED_MODEL_DIR)
        test(model, dataset_dict["test"], create_prompt_instruction)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune LLama-based model for AE Task")
    parser.add_argument("--test_only", type=bool, default=False, help="Test only")
    parser.add_argument("--zero_shot", type=bool, default=False, help="Test only")
    parser.add_argument("--force_reload_dataset", type=bool, default=False, help="")
    parser.add_argument("--zip_directory", type=str, default="./generated_effects.zip", help="")
    parser.add_argument("--include_gen_effects", type=bool, default=False, help="")
    params, unknown = parser.parse_known_args()
    main(params)