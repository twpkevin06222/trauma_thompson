import torch
from torchcodec.decoders import VideoDecoder

from transformers import AutoVideoProcessor, AutoModel
from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor
import numpy as np

model_name = "facebook/vjepa2-vitl-fpc16-256-ssv2"
# model = AutoModel.from_pretrained(model_name).to("cuda")
# processor = AutoVideoProcessor.from_pretrained(model_name)
processor = VJEPA2VideoProcessor.from_pretrained(model_name)
model = VJEPA2ForVideoClassification.from_pretrained(model_name).to("cuda")
model.config.num_labels = 124

total_params = sum(param.numel() for param in model.parameters())
print(f"Total number of parameters: {total_params}")
# 1. Freeze the pretrained VJEPA2 backbone (encoder + predictor)
for param in model.vjepa2.parameters():
    param.requires_grad = False
for param in model.pooler.parameters():
    param.requires_grad = False
# 2. Unfreeze only the classification head
for param in model.classifier.parameters():
    param.requires_grad = True


for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable:", name)

print(model.classifier.out_features)
# video_url = "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/holding_phone.mp4"
# vr = VideoDecoder(video_url)
# frame_idx = np.arange(
#     0, 32
# )  # choosing some frames. here, you can define more complex sampling strategy
# video_frames = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
# inputs = processor(video_frames, return_tensors="pt").to(model.device)

# with torch.no_grad():
#     outputs = model(**inputs)
#     print(outputs)
#     logits = outputs.logits

# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])
