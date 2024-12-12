import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from trl import SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from datasets import load_dataset, Dataset, DatasetDict
import os
import numpy as np
import random
import zipfile
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
NUM_CHOICES = 5
TEST_PERCENTAGE = 0.3
DATASET_DIR = "./dataset"

def process_zip(zip_path):
    # Temporary directory for extraction
    extract_dir = "./temp"

    # Extract ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Arrays to hold images and action verbs
    images = []
    action_verbs = []

    # Walk through extracted directories
    for root, _, files in os.walk(extract_dir):
        action_verb = os.path.basename(root)  # Folder name is the action verb
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):  # Check for image files
                file_path = os.path.join(root, file)
                try:
                    # Load image and append to the list
                    img = Image.open(file_path)
                    images.append(img)
                    action_verbs.append(action_verb)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    # Convert action verbs list to a NumPy array
    action_verbs_array = np.array(action_verbs, dtype=str)
    return images, action_verbs_array

def get_dataset(zip_dir):
  if not os.path.exists(DATASET_DIR):
    dataset = load_dataset("sled-umich/Action-Effect", trust_remote_code=True)
    dataset_imgs = [img for img_list in dataset["ActionEffect"]["positive_image_list"] for img in img_list]
    verb_nouns = np.array(dataset["ActionEffect"]["verb noun"], dtype=str)
    dataset_correct_verb_nouns = np.repeat(verb_nouns, [len(img_list) for img_list in dataset["ActionEffect"]["positive_image_list"]])

    gen_imgs, gen_correct_verb_nouns = process_zip(zip_dir)

    img_array = dataset_imgs + gen_imgs
    correct_verb_nouns = np.concatenate([dataset_correct_verb_nouns, gen_correct_verb_nouns])

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


dataset_dict = get_dataset("generated_images.zip")
train_dataset = dataset_dict["train"]
# eval_dataset = dataset_dict["test"]
eval_dataset = train_dataset



system_message = """You are a helpful assistant. You are shown an image. Select the number corresponding to the correct action that would cause the effect depicted in the image."""
#system_message = """You are a helpful assistant. You are shown an image. Select the number corresponding to the correct action that would cause the effect depicted in the image."""

def format_data(sample):
    question_choices = ''
    for j in range(len(sample['verb_noun_choices'])):
        question_choices += f"{j + 1}. {sample['verb_noun_choices'][j]}\n"
        return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["images"],
                },
                {
                    "type": "text",
                    # "text": sample["query"],
                    "text": f"{NUM_CHOICES} choices: {question_choices}"
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(sample["correct_indices"] + 1)}],
        },
    ]

# def format_data(sample):
    # question_choices = ''
    # for j in range(len(sample['verb_noun_choices'])):
    #     question_choices += f"{j + 1}. {sample['verb_noun_choices'][j]}\n"
    #     return [
    #     {
    #         "role": "system",
    #         "content": [{"type": "text", "text": system_message}],
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": sample["images"],
    #             },
    #             {
    #                 "type": "text",
    #                 # "text": sample["query"],
    #                 "text": f"{NUM_CHOICES} choices: {question_choices}"
    #             },
    #         ],
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [{"type": "text", "text": str(sample["correct_indices"] + 1)}],
    #     },
    # ]


train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]

from qwen_vl_utils import process_vision_info


def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample, tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

model_id = "Qwen/Qwen2-VL-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, device_map="cuda:0", torch_dtype=torch.bfloat16,
)
processor = Qwen2VLProcessor.from_pretrained(model_id)

adapter_path = "/nfs/turbo/coe-ahowens-nobackup/chfeng/qwen2-7b-instruct-trl-sft-action-effect/checkpoint-177"
model.load_adapter(adapter_path)


# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch


correct = 0
count = 0
for i in tqdm(range(len(eval_dataset))):
    output = generate_text_from_sample(model, processor, eval_dataset[i])
    label = eval_dataset[i][-1]['content'][0]['text']
    if output == label:
        correct += 1
    count += 1

print(correct / count)