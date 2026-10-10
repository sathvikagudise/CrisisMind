import re
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer

from backend.api.config import IMG_SIZE, TEXT_MAX_LEN, TEXT_MODEL_NAME

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    return _tokenizer


def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = IMG_TRANSFORM(image)
    return tensor.unsqueeze(0)


def tokenize_text(text: str):
    cleaned = clean_tweet(text)
    tokenizer = _get_tokenizer()
    encoded = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=TEXT_MAX_LEN,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]
