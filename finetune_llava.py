MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
TEST_PERCENTAGE = 0.3
NUM_CHOICES = 5
DATASET_DIR = "./dataset"
FINETUNED_MODEL_DIR = "./outputs"
SEED = 595
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import random
import argparse
import os
import json
from sklearn.model_selection import train_test_split
from huggingface_hub import login
from PIL import Image
import zipfile
import shutil
import wandb


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    SEED = 595
    set_seed(SEED)

    hugging_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    print(hugging_token)
    login(hugging_token)

    print("LOGIN DONE")

def process_zip(zip_path):
    extract_dir = "./temp"
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    images = []
    action_verbs = []
    
    for root, _, files in os.walk(extract_dir):
        action_verb = os.path.basename(root)  # Folder name is the action verb
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    images.append(img)
                    action_verbs.append(action_verb)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")

    action_verbs_array = np.array(action_verbs, dtype=str)
    return images, action_verbs_array

def get_dataset(params):
  if not os.path.exists(DATASET_DIR) or params.force_reload_dataset:
    dataset = load_dataset("sled-umich/Action-Effect", trust_remote_code=True)
    dataset_imgs = [img for img_list in dataset["ActionEffect"]["positive_image_list"] for img in img_list]
    verb_nouns = np.array(dataset["ActionEffect"]["verb noun"], dtype=str)
    dataset_correct_verb_nouns = np.repeat(verb_nouns, [len(img_list) for img_list in dataset["ActionEffect"]["positive_image_list"]])

    if params.include_gen_images:
        gen_imgs, gen_correct_verb_nouns = process_zip(params.zip_directory)
        
        img_array = dataset_imgs + gen_imgs
        correct_verb_nouns = np.concatenate([dataset_correct_verb_nouns, gen_correct_verb_nouns])
    else:
        img_array = dataset_imgs
        correct_verb_nouns = dataset_correct_verb_nouns
    verb_noun_choices = []
    correct_indices = []
    for i in range(len(img_array)):
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

    train_img, test_img, train_correct_verb_nouns, test_correct_verb_nouns, \
    train_verb_noun_choices, test_verb_noun_choices, train_correct_indices, test_correct_indices = train_test_split(
        img_array, correct_verb_nouns, verb_noun_choices, correct_indices, test_size=TEST_PERCENTAGE, random_state=42
    )

    train_dict = {
        "images": train_img,
        "correct_verb_nouns": train_correct_verb_nouns,
        "verb_noun_choices": train_verb_noun_choices,
        "correct_indices": train_correct_indices,
    }
    test_dict = {
        "images": test_img,
        "correct_verb_nouns": test_correct_verb_nouns,
        "verb_noun_choices": test_verb_noun_choices,
        "correct_indices": test_correct_indices,
    }

    # Convert to datasets.Dataset objects
    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })
    dataset_dict.save_to_disk(DATASET_DIR)
  else:
    print("Dataset already exists. Skipping preprocessing.")
    dataset_dict = DatasetDict.load_from_disk(DATASET_DIR)
  
  return dataset_dict


def load_llava(lora_config):
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )

    model = LlavaForConditionalGeneration.from_pretrained(
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

    return model


class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        labels = []
        correct_answer_indices = []
        for example in examples:
            messages = example
            correct_answer_indices.append(messages["correct_indices"] + 1)
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(messages["images"])

        batch = self.processor(images, texts, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Handle padding

        correct_answer_tokens = [
            self.processor.tokenizer.encode(str(index), add_special_tokens=False)[0]
            for index in correct_answer_indices
        ]
        correct_answer_tokens_tensor = torch.tensor(correct_answer_tokens, dtype=torch.long).unsqueeze(1)

        labels = torch.cat([labels, correct_answer_tokens_tensor], dim=1)
    
        batch["labels"] = labels
        return batch

def get_processor_tokenizer():
    template = """USER: You are a helpful assistant. You are shown an image. Select the correct action that would cause the effect depicted in the image.
    <image>
    Choices:
    {% for ind in range(messages["verb_noun_choices"]|length) %}
    {{ ind + 1 }}: {{ messages["verb_noun_choices"][ind] }}
    {% endfor %}
    Answer with one number from 1 to {{ messages["verb_noun_choices"]|length }}.
    ASSISTANT: """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.chat_template = template
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer = tokenizer
    processor.patch_size = None
    processor.vision_feature_select_strategy = None
    
    return processor, tokenizer

def train_model(
    dataset_dict,
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules="all-linear",
    num_epochs=3,
    lr=1.4e-5,
    per_device_train_batch_size=8,
    gradient_acc_steps=1,
    logging_steps=10,
    max_grad_norm=1.0,
    max_seq_length=512
    ):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )

    processor, tokenizer = get_processor_tokenizer()

    data_collator = LLavaDataCollator(processor)


    train_dataset = dataset_dict["train"]
    print("train shape:", train_dataset.shape)
    eval_dataset = train_dataset
    
    model = load_llava(lora_config)

    training_arguments = SFTConfig(
        run_name="finetune-llava",
        output_dir="./results",
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_acc_steps,
        logging_steps=logging_steps,
        max_grad_norm=max_grad_norm,
        max_seq_length=max_seq_length,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb",
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",  # need a dummy field
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    print("STARTING TRAIN")
    trainer.train()

    return trainer, model, processor


def save_model(trainer, model, output_dir):
    # Handle saving model in case of distributed/parallel training
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    model_to_save.model.save_pretrained(output_dir)

    # # Load the LoRA configuration from the saved model directory
    lora_config = LoraConfig.from_pretrained(output_dir)

    # # Reapply the LoRA configuration to the model
    model = get_peft_model(model, lora_config)

    return model


def clear_directory(dir):
    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Remove file or symbolic link
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directory

def test(model, dataset, processor, max_new_tokens=1, predictions_dir="./predictions"):

    total = 0
    num_correct = 0
    print("test size:", dataset.shape)
    out = {}
    count = 0
    os.makedirs(predictions_dir, exist_ok=True)
    clear_directory(predictions_dir)

    for example in dataset:
        messages = example
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image = messages["images"]
        batch = processor([image], [text], return_tensors="pt", padding=True).to("cuda")
        # pixel_values = batch["pixel_values"]
        # input_ids = batch["input_ids"]
        # inputs = processor(prompts, images=[image1, image2], padding=True, return_tensors="pt")
        output = model.generate(**batch, max_new_tokens=max_new_tokens)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        response = generated_text[0].split("ASSISTANT:")[-1].strip()
        
        if response == str(messages["correct_indices"] + 1):
            num_correct += 1
        else:
            file_path = os.path.join(predictions_dir, f"image{count}.{image.format}")
            image.save(file_path)
            out[f"image{count}.{image.format}"] = ((messages["verb_noun_choices"][int(response) - 1]) if int(response) - 1 in range(NUM_CHOICES) else "N/A",
                                                   messages["correct_verb_nouns"])          
        total += 1
    
    out["accuacy"] = {num_correct/total}
    
    with open(os.path.join(predictions_dir, "out.json"), "w") as file:
        json.dump(out, file, indent=4)
    
    print(f"Accuracy: {num_correct/total}")
            
def load_llava_trained_lora(output_dir):
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
    model = LlavaForConditionalGeneration.from_pretrained(
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
        wandb.login()

        run = wandb.init(
            project='Finetuning LLaVA-v1.5',
            job_type="training",
            anonymous="allow"
        )
        trainer, model, processor = train_model(dataset_dict,r=16, lora_alpha=32, lora_dropout=0.3, num_epochs=2, max_grad_norm=0.75)
        model = save_model(trainer, model, FINETUNED_MODEL_DIR)

        test(model, dataset_dict["test"], processor)
        
    else:
        model = load_llava_trained_lora(FINETUNED_MODEL_DIR)
        processor, _ = get_processor_tokenizer()
        test(model, dataset_dict["test"], processor)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune LLaVA-based model for Action effect prediction")
    parser.add_argument("--test_only", type=bool, default=False, help="Test only")
    parser.add_argument("--force_reload_dataset", type=bool, default=False, help="")
    parser.add_argument("--zip_directory", type=str, default="./generated_images.zip", help="")
    parser.add_argument("--include_gen_images", type=bool, default=False, help="")
    params, unknown = parser.parse_known_args()
    main(params)