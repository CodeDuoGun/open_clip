import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="./logs/tongue_vitb32/xxx/checkpoints/epoch_10.pt",
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("test.jpg")).unsqueeze(0)
texts = tokenizer([
    "舌质淡红，薄白苔",
    "舌红，黄腻苔",
    "舌胖大，有齿痕",
])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(probs)