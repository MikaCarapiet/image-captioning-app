"""File for automating URL captions"""
from io import BytesIO
import requests
from PIL import Image
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained transformer and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL of the page to scrape
URL = "https://en.wikipedia.org/wiki/National_Basketball_Association"

# Download the page
response = requests.get(URL,timeout=60)

#Parse the page with Beautiful Soup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all the img elements
img_elements = soup.find_all('img')

# Open a file to write the captions
with open("captions.txt", "w", encoding="utf-8") as caption_file:
    # Iterate over each img element
    for img_element in img_elements:
        img_url = img_element.get("src")

        # Skip if the image is SVG or most likely an icon
        if 'svg' in img_url or '1x1' in img_url:
            continue

        # Correct the URL if it's malformed
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http://') or not img_url.startswith('https://'):
            continue

        try:
            # Download the image
            response = requests.get(img_url,timeout=60)

            # Convert the image data to a PIL image
            raw_image = Image.open(BytesIO(response.content)).convert('RGB')
            if raw_image.size[0] * raw_image.size[1] < 400: # Skip very small images
                continue
            # Process the image
            inputs = processor(raw_image, return_tensors="pt")
            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            captions = processor.decode(out[0],skip_special_tokens=True)

            # Write the caption to the file, prepended by the image URL
            caption_file.write(f"{img_url}: {captions}\n")
        except ImportError as e:
            print(f"Error processing image {img_url}: {e}")
            continue
