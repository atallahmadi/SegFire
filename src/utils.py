import os
import math
import json
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
    Normalize,
    Compose,
    Resize
)


class LandsatDataset(Dataset):
    def __init__(self, dataset, size= 256, do_augmentation=True):
        self.paths = dataset
        self.size = size
        self.do_augmentation = do_augmentation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        label_path = get_label_path(image_path)

        image, label = read_image(image_path).float() / 255.0, read_image(label_path).float() / 255.0
        image, label = self.__augment__(image, label)

        return image, label, image_path

    def __augment__(self, image, label):
        both = torch.cat([image, label], dim=0)  # Shape: [4, 256, 256]

        if self.do_augmentation:
            both = Compose([
                Resize((self.size, self.size)),
                RandomRotation(degrees=45),
                RandomVerticalFlip(),
                RandomHorizontalFlip()
                ])(both)
            
            image, label = torch.split(both, [3, 1], dim=0)  # Shape: [3, 256, 256], [1, 256, 256]     
            image = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)(image)
            # image = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
            label = label.squeeze().long()

            # Shape: [3, 256, 256], [256, 256]       
            return image, label
        
        else:
            both = Resize((self.size, self.size))(both)
            
            image, label = torch.split(both, [3, 1], dim=0)  # Shape: [3, 256, 256], [1, 256, 256]     
            image = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
            label = label.squeeze().long()

            # Shape: [3, 256, 256], [256, 256]       
            return image, label


def get_image_path(datapath, type='png', limit=None, seed=1):
    image_paths = [str(path) for path in list(Path(datapath).glob(f'./images/*.{type}'))]
    if limit is not None:
        random.seed(seed)
        random.shuffle(image_paths)
        image_paths= image_paths[:limit]

    return sorted(image_paths)


def get_label_path(image_path):
    split = image_path.rfind('_')
    label_path = f"{image_path[:split]}_voting{image_path[split:]}".replace('images', 'masks')
    
    return label_path


def plotter(dataloader):
    image, label, path = next(iter(dataloader))
    print(f"Batched Image Shape: {image.shape}") 
    print(f"Batched Label Shape: {label.shape}")

    idx = random.randint(0, image.shape[0] - 1)

    og = read_image(path[idx]).permute(1, 2, 0).numpy()
    
    image = image[idx]
    label = label[idx]

    # Convert the image tensor from (C, H, W) to (H, W, C) and numpy
    image_np = image.permute(1, 2, 0).numpy()
    label_np = label.squeeze().numpy()

    print(f"Image Shape: {image.shape}") 
    print(f"Label Shape: {label.shape}")

    print(f"Image Min: {image_np.min().item()}")
    print(f"Image Max: {image_np.max().item()}")

    print(f"Label Min: {label_np.min().item()}")
    print(f"Label Max: {label_np.max().item()}")

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(og)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image_np)
    axes[1].set_title("Transformed Image")
    axes[1].axis("off")

    axes[2].imshow(label_np)
    axes[2].set_title("Label")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()


def plotter_results(images, masks, predictions, path, num_samples=3):
    """
    Plots the images, predictions, and ground truths side by side.
    """
    num_batchs = len(images)
    num_qurry = images[0].shape[0] 

    count = min(num_qurry, num_samples)
    idx = random.randint(0, num_batchs - 1)

    for i in range(0, count):
        img = images[idx][i].permute(1, 2, 0).numpy()
        pred = predictions[idx][i].numpy()
        mask = masks[idx][i].numpy()

        _, axes = plt.subplots(1, 4, figsize=(15, 5))
        og = read_image(path[idx][i]).permute(1, 2, 0).numpy()

        print('Showing: ', os.path.basename(path[idx][i]))
        axes[0].imshow(og)
        axes[0].set_title(f"Original Image")
        axes[0].axis("off")

        axes[1].imshow(mask)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].set_title(f"Transformed Image")
        axes[2].axis("off")

        axes[3].imshow(pred)
        axes[3].set_title("Prediction")
        axes[3].axis("off")
        
        plt.tight_layout()
        plt.show()


def plotter_history(json_path):
    with open(json_path, 'r') as file:
        history = json.load(file)
    
    epochs = history["epoch"]
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()
       
    # Plot precision, recall, and F1-score
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history["precision"], label="Precision", marker='o')
    plt.plot(epochs, history["recall"], label="Recall", marker='s')
    plt.plot(epochs, history["f1_score"], label="F1 Score", marker='^')
    plt.plot(epochs, history["iou"], label="IoU", marker='*')
    plt.plot(epochs, history["miou"], label="mIoU", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision, Recall, F1 Score, IoU, and mIoU Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()


def seed_everything(seed=42):
    random.seed(seed)  # Python’s built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU


def replace_nan(value):
    return 0.0 if math.isnan(value) or np.isnan(value) else value


def compute_metrics(conf_matr):
    predicted_sum = conf_matr.sum(axis=0)  # Sum along columns
    gt_sum = conf_matr.sum(axis=1)  # Sum along rows
    diag = np.diag(conf_matr)

    precision = np.true_divide(
        diag, predicted_sum, np.full(diag.shape, np.nan), where=predicted_sum != 0
    )
    recall = np.true_divide(
        diag, gt_sum, np.full(diag.shape, np.nan), where=gt_sum != 0
    )
    num = 2 * (precision * recall)
    den = precision + recall
    f1 = np.true_divide(num, den, np.full(num.shape, np.nan), where=den != 0)

    intersection = diag
    union = predicted_sum + gt_sum - diag
    iou = np.true_divide(intersection, union, np.full(intersection.shape, np.nan), where=union != 0)
    
    return precision, recall, f1, iou


def test(name, model, test_, output_dir, th=0.5):
    model.eval()
    gts, prd, img, paths = [], [], [], []
    
    with torch.no_grad():
        for batch in test_:
            images, masks, image_paths = batch[0].cuda().float(), batch[1].cuda().long(), batch[2]
            
            outputs = model(pixel_values=images)
            logits = nn.functional.interpolate(outputs.logits, size=masks.shape[-1], mode='bilinear', align_corners=False)
            probability = torch.softmax(logits, dim=1)
            predictions = (probability[:, 1, :, :] > th).long()

            paths.append(image_paths)
            img.append(images.cpu())
            prd.append(predictions.cpu())
            gts.append(masks.cpu())

    prd_tensor = torch.cat(prd).numpy().flatten()
    gts_tensor = torch.cat(gts).numpy().flatten()

    precision, recall, f1, iou = compute_metrics(confusion_matrix(gts_tensor, prd_tensor))
    # print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nIoU: {iou}\nmIoU: {np.mean(iou)}")
    print(f"Precision: {np.mean(precision)}\nRecall: {np.mean(recall)}\nF1 Score: {np.mean(f1)}\nmIoU: {np.mean(iou)}")

    # Save test metrics as JSON
    test_metrics = {
        "precision": float(precision[1]),    # Class 1 precision
        "recall": float(recall[1]),          # Class 1 recall
        "IoU": float(iou[1]),                # Class 1 IoU
        "mIoU": float(np.mean(iou)),         # Mean IoU
        "f1_score": float(f1[1])             # Class 1 F1 Score
    }

    with open(f"{output_dir}/{name}/test.json", 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    return img, gts, prd, paths


def train_and_validate(name, model, train_, val_, loss_, optimizer, scheduler_type, epochs, delta, early_stop, output_dir):
    no_improvement = 0
    best_metric_value = float('-inf')

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "iou": [],
        "miou": []
    }

    # Initialize scheduler based on type
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epochs)
    else:
        pass

    for epoch in range(epochs):
        print(20*'------')

        if no_improvement >= early_stop:
            break
        
        model.train()
        train_loss = []
        train_loader = tqdm(train_, desc=f"Epoch {epoch+1}/{epochs} - Training")
        for batch in train_loader:
            images, masks = batch[0].cuda().float(), batch[1].cuda().long()

            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            logits = nn.functional.interpolate(outputs.logits, size=masks.shape[-1], mode='bilinear', align_corners=False)
            loss = loss_(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_loader.set_postfix({"Loss": loss.item()})
            del images, masks, outputs, logits
            torch.cuda.empty_cache()


        model.eval()
        val_loss, total_confusion_matrix = [], None
        val_loader = tqdm(val_, desc=f"Epoch {epoch+1}/{epochs} - Validation")
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch[0].cuda().float(), batch[1].cuda().long()

                outputs = model(pixel_values=images)
                logits = nn.functional.interpolate(outputs.logits, size=masks.shape[-1], mode='bilinear', align_corners=False)
                loss = loss_(logits, masks)
                probability = torch.softmax(logits, dim=1)
                predictions = (probability[:, 1, :, :] > 0.5).long()

                # Calculate confusion matrix per batch
                prd_tensor = predictions.cpu().numpy().flatten()
                gts_tensor = masks.cpu().numpy().flatten()
                batch_matrix = confusion_matrix(gts_tensor, prd_tensor, labels=[0, 1])

                if total_confusion_matrix is None:
                    epoch_matrix = batch_matrix
                else:
                    epoch_matrix += batch_matrix
                
                val_loss.append(loss.item())
                val_loader.set_postfix({"Loss": loss.item()})

                del images, masks, outputs, logits, probability, predictions
                torch.cuda.empty_cache()
        
        precision, recall, f1, iou = compute_metrics(epoch_matrix)

        print(f'Epoch: {epoch+1}/{epochs} | Train_loss: {np.mean(train_loss)} | Val_loss: {np.mean(val_loss)}')
        print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nIoU: {iou}\nmIoU: {np.mean(iou)}")

        current_metric_value = replace_nan(precision[1]) #for class 1 aka fire
        if ((current_metric_value - best_metric_value) > delta):
            os.makedirs(f"{output_dir}/{name}", exist_ok=True)
            model.save_pretrained(f"{output_dir}/{name}")
            no_improvement = 0
            print(f'Did improve from {best_metric_value} to {current_metric_value}')
            print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")
            best_metric_value = current_metric_value
        else:
            no_improvement += 1
            print(f'Early stopping after {no_improvement}/{early_stop} - Did not improve from {best_metric_value}')
            print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(np.mean(val_loss))  # Adjust based on val loss
        else:
            # scheduler.step()  # Step normally
            pass

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(np.mean(train_loss)))
        history["val_loss"].append(float(np.mean(val_loss)))
        history["precision"].append(replace_nan(float(precision[1])))
        history["recall"].append(replace_nan(float(recall[1])))
        history["f1_score"].append(replace_nan(float(f1[1])))
        history["iou"].append(replace_nan(float(iou[1])))
        history["miou"].append(float(np.mean(iou)))
        with open(f"{output_dir}/{name}/history.json", 'w') as f:
            json.dump(history, f, indent=4)
