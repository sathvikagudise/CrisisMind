import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import DistilBertModel


class ImageFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path=None, freeze=True):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        if checkpoint_path is None:
            new_state = {k: v for k, v in efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1).state_dict().items()
                if not k.startswith("classifier.")}
        else:
            state = torch.load(checkpoint_path, map_location="cpu")
            new_state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        self.backbone.load_state_dict(new_state, strict=False)
        self.backbone.classifier = nn.Identity()
        self.out_dim = in_features
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class TextFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path=None, freeze=True):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.out_dim = self.distilbert.config.hidden_size
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            distilbert_state = {k.replace("distilbert.", ""): v
                               for k, v in state.items()
                               if k.startswith("distilbert.")}
            self.distilbert.load_state_dict(distilbert_state, strict=False)
        if freeze:
            for p in self.distilbert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :]
        return cls_embed


class ConcatFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, num_classes=3, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(img_dim + txt_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, img_feat, txt_feat):
        fused = torch.cat([img_feat, txt_feat], dim=1)
        x = self.fc1(fused)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatedFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, num_classes=3, dropout=0.3):
        super().__init__()
        self.img_dim = img_dim
        self.txt_dim = txt_dim

        self.gate_net = nn.Sequential(
            nn.Linear(img_dim + txt_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(img_dim + txt_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, img_feat, txt_feat):
        concat = torch.cat([img_feat, txt_feat], dim=1)
        gates = self.gate_net(concat)
        g_img = gates[:, 0:1]
        g_txt = gates[:, 1:2]
        gated_img = g_img * img_feat
        gated_txt = g_txt * txt_feat
        fused = torch.cat([gated_img, gated_txt], dim=1)
        x = self.fc1(fused)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, gates


class GatedFusionWide(nn.Module):
    def __init__(self, img_dim=1280, txt_dim=768, num_classes=3, dropout=0.3, gate_dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Dropout(gate_dropout),
            nn.Linear(img_dim + txt_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(img_dim + txt_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, img_feat, txt_feat):
        concat = torch.cat([img_feat, txt_feat], dim=1)
        gates = self.gate_net(concat)
        g_img, g_txt = gates[:, 0:1], gates[:, 1:2]
        gated = torch.cat([g_img * img_feat, g_txt * txt_feat], dim=1)
        x = self.bn1(self.fc1(gated))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x, gates


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
