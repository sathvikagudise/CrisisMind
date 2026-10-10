import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertConfig
from backend.training.config import TEXT_MODEL_NAME


def build_text_model(num_classes=3, freeze_bert=False):
    config = DistilBertConfig.from_pretrained(TEXT_MODEL_NAME)
    config.num_labels = num_classes
    model = DistilBertForSequenceClassification.from_pretrained(TEXT_MODEL_NAME, config=config)

    if freeze_bert:
        for param in model.distilbert.parameters():
            param.requires_grad = False

    return model


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
