import streamlit as st
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from test import iou_fn, focal_fn, dice_fn
import numpy as np
import time

## Load model
model = smp.Unet(
    encoder_name='resnet50', 
    encoder_weights='imagenet', 
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load("models/DFUCmodelV0.pt"))


## GUI
st.title("Diabetic Foot Ulcer (DFU) Classifier")
img_uploader = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])
mask_uploader = st.file_uploader("Upload a mask file of the previous DFU image", type=["jpg", "png", "jpeg"])


if img_uploader is not None:
    if mask_uploader is not None:
        image = Image.open(img_uploader).convert("RGB")
        mask = Image.open(mask_uploader).convert("L")
        st.image(img_uploader, caption="Image uploaded", use_container_width=True)
        st.image(mask_uploader, caption="Mask uploaded", use_container_width=True)

        img_transforms = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        def preprocess(image, mask):
            image = np.array(image)
            mask = np.array(mask)

            # converting mask from 0-255 to binary
            mask = (mask > 127).astype(np.float32)
            augmented = img_transforms(image = image, mask = mask)
            img_tensor = augmented['image']
            mask_tensor = augmented['mask'].unsqueeze(0).float()

            return img_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)

        img_tensor, mask_tensor = preprocess(image, mask)

        if st.button("Detect DFU"):
            start_time = time.time()
            preds = model(img_tensor)    
            loss = 0.5 * dice_fn(preds, mask_tensor) + 0.5 * focal_fn(preds, mask_tensor)
            iou = iou_fn(preds, mask_tensor)
            end_time = time.time() - start_time

            pred_mask = torch.sigmoid(preds)
            pred_mask = (pred_mask > 0.5).float()
            pred_mask_np = pred_mask.squeeze().cpu().numpy()
            pred_mask_np = (pred_mask_np * 255).astype(np.uint8)

            st.write(f"IOU: {iou:.2f}% | Loss: {loss:.2f} | Time: {end_time:.2f}s")
            st.image(pred_mask_np, caption="Predicted DFU Mask", use_column_width=True)
    else:
        st.error("Please upload a mask file.")
else:
    st.error("Please upload an image file")