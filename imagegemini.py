
from pathlib import Path
import google.generativeai as genai


genai.configure(api_key="AIzaSyCBNGrGObQlMcvQZljqcQV0g5zLmd5ZUEU")

# Set up the model
generation_config = {
  "temperature": 0.4,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Validate that an image is present
if not (img := Path("image0.jpeg")).exists():
  raise FileNotFoundError(f"Could not find image: {img}")

image_parts = [
  {
    "mime_type": "image/jpeg",
    "data": Path("image0.jpeg").read_bytes()
  },
  {
    "mime_type": "image/jpeg",
    "data": Path("image1.jpeg").read_bytes()
  },
]

prompt_parts = [
  "Please classify the following images based on whether they contain humans or non-human subjects.\n\n1. ",
  image_parts[0],
  " [Provide a brief description of the content of Image 1]\n2. ",
  image_parts[1],
  ": [Provide a brief description of the content of Image 2]\n3. a fake image [Provide a brief description of the content of Image 3]\nFor each image, respond with 'Human' if the image contains one or more humans, and 'Non-Human' if the image does not contain any humans.\nthe main job is to identify whether the image is REAL or FAKE\n",
]

response = model.generate_content(prompt_parts)
print(response.text)