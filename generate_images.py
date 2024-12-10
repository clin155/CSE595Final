from time import sleep
import openai
import os
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset, Dataset, DatasetDict
client = openai.OpenAI()
# Set your OpenAI API key
def create_output_folder(folder_name):
    """
    Create a folder if it doesn't exist.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def generate_and_save_images(prompt, output_folder, size="1024x1024", num_images=5, output_dir="generated_images"):
    """
    Generate images from a list of prompts using OpenAI's DALL-E API and save them.

    Args:
        prompts (list): List of text prompts describing the images.
        size (str): Desired size of the images ("256x256", "512x512", "1024x1024").
        num_images (int): Number of images to generate per prompt.
        output_dir (str): Root directory where images will be saved.
    """
    count = 0
    for p in prompt:
        print(f"Generating image for {p}")
        # Create a folder for this prompt
        prompt_folder = os.path.join(output_dir, output_folder)
        create_output_folder(prompt_folder)
        try:
            # Generate image from API
            response = client.images.generate(
                model="dall-e-3",
                prompt=p,
                size=size,
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            
            # Download the image
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))

            # Save the image
            image_filename = os.path.join(prompt_folder, f"image_{count}.png")
            image.save(image_filename)
            print(f"Saved: {image_filename}")
            sleep(1)
            count += 1

        except Exception as e:
            print(f"Error generating image for prompt '{prompt}'")
            continue

if __name__ == "__main__":
    template = "Generate an image showing this: "
    
    size = "1024x1024"
    num_images = 5  # Number of images per prompt
    output_dir = "generated_images"
    dataset = load_dataset("sled-umich/Action-Effect")["ActionEffect"]
    found = False
    for action_verb, effect_list in zip(dataset["verb noun"], dataset["effect_sentence_list"]):
        prompts = [template + e for e in effect_list[:5]]
        generate_and_save_images(prompts, action_verb, size, num_images, output_dir)
    
    # generate_and_save_images(prompts, size, num_images, output_dir)
