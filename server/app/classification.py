import torch
from transformers import AutoModelForImageClassification
import numpy as np
from torchvision import transforms

class Classification:
    def __init__(self, model, label2id, id2label, device='cpu'):
        self.label2id = label2id
        self.id2label = id2label
        self.pil_to_t_transform = transforms.Compose(
            [
            #  transforms.PILToTensor(),
             transforms.functional.to_tensor,
             transforms.Resize((224, 224))
             ])
        
        self.model = AutoModelForImageClassification.from_pretrained(
            model, 
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes = True, # provide this in case you'd like to fine-tune an already fine-tuned checkpoint
        ).to(device)

    def classify(self, data):
        data = self.pil_to_t_transform(data).unsqueeze(0)
        with torch.no_grad():
            return self.id2label[np.argmax(self.model(data).logits.cpu().numpy())]