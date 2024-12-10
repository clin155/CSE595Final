MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
NUM_CHOICES = 5
DATASET_DIR = "./dataset"
SEED = 595
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
from datasets import load_dataset, Dataset, DatasetDict
import argparse
import torch
import random
import shutil
import json
import os
from huggingface_hub import login

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)

hugging_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
print(hugging_token)
login(hugging_token)

print("LOGIN DONE")

from finetune_llava import get_dataset, get_processor_tokenizer, clear_directory

def test(model, dataset_dict, processor, max_new_tokens=1, predictions_dir="./zero_shot_predictions"):

    total = 0
    num_correct = 0
    print("test size:", dataset_dict["train"].shape[0] + dataset_dict["test"].shape[0])

    out = {}
    count = 0
    os.makedirs(predictions_dir, exist_ok=True)
    clear_directory(predictions_dir)

    for item in os.listdir(predictions_dir):
        item_path = os.path.join(predictions_dir, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Remove file or symbolic link
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directory

    for key in dataset_dict:
        dataset = dataset_dict[key]
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

def main(params):
    dataset_dict = get_dataset(params)
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    processor, _ = get_processor_tokenizer()
    
    test(model, dataset_dict, processor)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot LLaVA-based model for Action Effect Prediction")
    parser.add_argument("--force_reload_dataset", type=bool, default=False, help="")
    parser.add_argument("--zip_directory", type=str, default="./generated_images.zip", help="")
    parser.add_argument("--include_gen_images", type=bool, default=False, help="")
    params, unknown = parser.parse_known_args()
    main(params)