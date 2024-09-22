"""Image Captioning Script"""
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image
IMG_PATH = "plaza.jpg"
#conver it into an RGB format
image = Image.open(IMG_PATH).convert('RGB')

# Image captioning
TEXT = "the image of"
inputs = processor(images=image,text=TEXT, return_tensors="pt")

# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens to text
caption = processor.decode(outputs[0],skip_special_tokens=True)
# Print the caption
print(caption)
