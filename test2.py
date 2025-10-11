import segmentation_models_pytorch as smp
import torch
import time
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image

if __name__ == "__main__":
        torch.backends.cudnn.benchmark = True

        torch.set_float32_matmul_precision("medium")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = smp.Unet(
        encoder_name="resnet18",  
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,  # raw logits produced without activation function
        )


        ## STUFF TO INITIALISE
        model = model.to(device)
        loss_fn = smp.losses.DiceLoss(mode='binary')
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        img_transforms = tv.transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

        # Masks
        mask_transforms = tv.transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
        ])

        def iou_fn(y_pred, y_true):
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()
            inter = (y_pred * y_true).sum(dim=(1,2,3))
            union = y_pred.sum(dim=(1,2,3)) + y_true.sum(dim=(1,2,3))
            iou = (inter + 1e-6) / (union + 1e-6)
            return iou.mean().item() * 100

        ### PREPROCESSING IMAGES
        class DFUdataset(Dataset):
        
        # Initialise w/ directories
            def __init__(self, image_dir, mask_dir, img_transforms=None, mask_transforms=None):
                self.image_dir = image_dir
                self.mask_dir = mask_dir
                self.img_transform = img_transforms
                self.mask_transform = mask_transforms
                self.images = sorted(os.listdir(image_dir))
                self.masks = sorted(os.listdir(mask_dir))

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image_path = os.path.join(self.image_dir, self.images[idx])
                mask_path = os.path.join(self.mask_dir, self.masks[idx])

                image = Image.open(image_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")  # Grayscale 

                if self.img_transform:
                    image = self.img_transform(image)
                if self.mask_transform:
                    mask = self.mask_transform(mask)

                return image, mask
        

        # Directories addresses
        DATA_DIR = "dataset"
        dataset = DFUdataset(
        image_dir=os.path.join(DATA_DIR, "images"), 
        mask_dir=os.path.join(DATA_DIR, "masks"),
        img_transforms=img_transforms,
        mask_transforms=mask_transforms
        )

        # Train-test split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

        for images, masks in train_loader:
            print(images.shape, masks.shape)
            break

        scaler = torch.amp.GradScaler()


        ## TRAINING
        torch.manual_seed(42)
        epochs = 10

        for epoch in range(epochs):
            model.train()   
            running_loss, running_iou, correct, total = 0.0, 0.0, 0, 0
            start_time = time.time()

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    preds = model(X_batch)
                    loss = loss_fn(preds, y_batch)

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
                    test_loss = loss_fn(test_preds, y_batch)

                    running_test_loss += test_loss.item()
                    running_test_iou += iou_fn(test_preds, y_batch)

            torch.cuda.empty_cache()

            # timing epochs
            end_time = time.time()
            epoch_time = end_time - start_time
            
            print(f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {(running_loss/len(train_loader)):.2f} "
                f"Accuracy {(running_iou/len(train_loader)):.2f}% "
                f"Test Loss: {(running_test_loss/len(val_loader)):.2f}"
                f"Test Accuracy: {(running_test_iou/len(val_loader)):.2f}% "
                f"Time: {epoch_time:.2f}s")