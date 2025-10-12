import segmentation_models_pytorch as smp
import torch
import time
import torchvision as tv
import torchvision.transforms as transforms
import os
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path

img_transforms = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p = 0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def iou_fn(y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.5).float()
        
        if y_pred.ndim == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.ndim == 3:
             y_true = y_true.unsqueeze(1)

        inter = (y_pred * y_true).sum(dim=(1,2,3))
        union = y_pred.sum(dim=(1,2,3)) + y_true.sum(dim=(1,2,3)) - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        return iou.mean().item() * 100


### PREPROCESSING IMAGES
class DFUdataset(Dataset):
        
        # Initialise w/ directories
        def __init__(self, image_dir, mask_dir, img_transforms=None):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.img_transforms = img_transforms   
            self.images = sorted(os.listdir(image_dir))     
            self.masks = sorted(os.listdir(mask_dir))

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.masks[idx])

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # Grayscale 

            image = np.array(image)
            mask = np.array(mask)

            # converting mask from 0-255 to binary
            mask = (mask > 127).astype(np.float32)
            
            augmented = self.img_transforms(image = image, mask = mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()

            return image, mask
        

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="resnet50",  
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,  # raw logits produced without activation function
    )


    ## STUFF TO INITIALISE
    model = model.to(device)
    dice_fn = smp.losses.DiceLoss(mode='binary')
    focal_fn = smp.losses.FocalLoss('binary')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.3)

    # Directories addresses
    DATA_DIR = "dataset"
    dataset = DFUdataset(
        image_dir=os.path.join(DATA_DIR, "images"), 
        mask_dir=os.path.join(DATA_DIR, "masks"),
        img_transforms=img_transforms
    )

    # Train-test split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

    for images, masks in train_loader:
        print(images.shape, masks.shape)
        break

    scaler = torch.amp.GradScaler()


    ## TRAINING
    torch.manual_seed(42)
    best_iou = 0
    epochs = 30

    def save(model, epochs=epochs+1, iou=None):
        ## Saving the model
        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(exist_ok=True, parents=True)
        MODEL_SAVE_PATH = MODEL_PATH / "DFUCmodelV0.pt"
        print("Saving model")
        torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
    
    for epoch in range(epochs):
        model.train()   
        running_loss, running_iou, correct, total = 0.0, 0.0, 0, 0
        start_time = time.time()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                preds = model(X_batch)
                loss = 0.5 * focal_fn(preds, y_batch) + 0.5 * dice_fn(preds, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_iou += iou_fn(preds, y_batch)

        model.eval()
        running_test_loss, running_test_iou = 0.0, 0.0
        with torch.inference_mode():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                test_preds = model(X_batch)
                test_loss = 0.5 * focal_fn(test_preds, y_batch) + 0.5 * dice_fn(test_preds, y_batch)

                running_test_loss += test_loss.item()
                running_test_iou += iou_fn(test_preds, y_batch)

        torch.cuda.empty_cache()

        # timing epochs
        end_time = time.time()
        epoch_time = end_time - start_time

        avg_train_loss = running_loss / len(train_loader)
        avg_train_iou = running_iou / len(train_loader)
        avg_val_loss = running_test_loss / len(val_loader)
        avg_val_iou = running_test_iou / len(val_loader)

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            save(model)
       
        print(f"Epoch [{epoch+1}/{epochs}] | "
            f"Loss: {(running_loss/len(train_loader)):.2f} | "
            f"Accuracy: {(running_iou/len(train_loader)):.2f}% | "
            f"Test Loss: {(running_test_loss/len(val_loader)):.2f} | "
            f"Test Accuracy: {(running_test_iou/len(val_loader)):.2f}% | "
            f"Time: {epoch_time:.2f}s")
        
        scheduler.step(avg_val_iou)