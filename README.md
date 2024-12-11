# CSE595Final

How to run:
set HUGGINGFACE_HUB_TOKEN and WANDB_API_KEY environment variables
for fine tuning:
python finetune_llava.py 
    args: force_reload_dataset -> dataset is cached in DATASET_DIR, add this arg to refresh it 
        zip_directory -> location of the generated_images.zip (the zip of the chat-gpt generated images)
        include_gen_images -> whether to include the generated chatgpt images
        test_only -> only run inference no fine tuning
python finetune_llama.py
    args: all the same args except for test_only

python qwen-vl-7b-sft.py
    args: all the same args except for test_only