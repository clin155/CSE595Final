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
eval_dataset = train_dataset


#system_message = """You are a helpful assistant. You are shown an image. Select the correct action that would cause the effect depicted in the image."""
system_message = """You are a helpful assistant. You are shown an image. Select the number corresponding to the correct action that would cause the effect depicted in the image."""

# def format_data(sample):
#     return [
#         {
#             "role": "system",
#             "content": [{"type": "text", "text": system_message}],
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": sample["images"],
#                 },
#                 {
#                     "type": "text",
#                     # "text": sample["query"],
#                     "text": f"{NUM_CHOICES} choices: {', '.join(sample['verb_noun_choices'])}"
#                 },
#             ],
#         },
#         {
#             "role": "assistant",
#             "content": [{"type": "text", "text": str(sample["correct_indices"] + 1)}],
#         },
#     ]

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


train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]

model_id = "Qwen/Qwen2-VL-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, device_map="cuda:0", torch_dtype=torch.bfloat16, quantization_config=bnb_config
)
processor = Qwen2VLProcessor.from_pretrained(model_id)


# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=64,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config)

# Print trainable parameters
peft_model.print_trainable_parameters()


# Configure training arguments
training_args = SFTConfig(
    output_dir="/nfs/turbo/coe-ahowens-nobackup/chfeng/qwen2-7b-instruct-trl-sft-action-effect",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=16,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=50,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=50,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    report_to="wandb",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=1024  # Maximum sequence length for input
)

training_args.remove_unused_columns = False  # Keep unused columns in dataset


import wandb

wandb.init(
    project="qwen2-7b-instruct-trl-sft-action-effect",  # change this
    name="qwen2-7b-instruct-trl-sft-action-effect",  # change this
    config=training_args,
)

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


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

trainer.train()