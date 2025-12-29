import os
import sys
import pandas as pd
import numpy as np
from PIL import Image

from sympy.sets.sets import false
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


# ========================
# Focal Loss Class
# ========================
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs, targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )

# ========================
# Dataset preparation
# ========================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels


class RetinaInferenceDataset(Dataset):
    """Dataset for inference without labels"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.image_files[idx]


# ========================
# build model
# ========================
def build_model(backbone="resnet18", num_classes=3):

    if backbone == "resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model


# ========================
# model training and val
# ========================
def train_one_backbone(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir, 
                       epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints",pretrained_backbone=None,
                       freeze_backbone=False, loss='focal'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # transforms - separate for train and val/test
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, train_transform)
    val_ds   = RetinaMultiLabelDataset(val_csv, val_image_dir, val_test_transform)
    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # model
    model = build_model(backbone, num_classes=3).to(device)


    for p in model.parameters():
        p.requires_grad = False

    # Freeze backbone if requested
    if freeze_backbone:
        if backbone == "resnet18":
            # Freeze all layers except the classifier (fc)
            for param in model.fc.parameters():
                param.requires_grad = True
        elif backbone == "efficientnet":
            # Freeze all layers except the classifier
            for param in model.classifier.parameters():
                param.requires_grad = True
        print(f"[{backbone}] Backbone frozen. Only classifier will be trained.")
    else:
        # All parameters trainable
        for p in model.parameters():
            p.requires_grad = True
    
    # loss & optimizer
    if loss == 'focal':
        focal_alpha = 0.5
        focal_gamma = 1.5
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"[{backbone}] Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    elif loss == 'bce-balanced':
        # Calculated from training data
        pos_weight = [0.5474, 3.9080, 4.6338]
        pos_weight_tensor = torch.tensor(pos_weight).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"[{backbone}] Using BCEWithLogitsLoss with class balancing, pos_weight: {pos_weight}")
    elif loss == 'bce-logits':
        criterion = nn.BCEWithLogitsLoss()
        print(f"[{backbone}] Using BCEWithLogitsLoss")
    else:
        raise ValueError(f"Unknown loss type: {loss}. Choose from 'focal', 'bce-logits', or 'bce-balanced'")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict)
    

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{backbone}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model for {backbone} at {ckpt_path}")

    # ========================
    # testing
    # ========================
    if epochs != 0:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"[{backbone}] No training performed. Saving initial model as best model.")
        torch.save(model.state_dict(), ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = torch.tensor(np.array(y_true))
    y_pred = torch.tensor(np.array(y_pred))

    disease_names = ["DR", "Glaucoma", "AMD"]

    for i, disease in enumerate(disease_names):  #compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro",zero_division=0)
        recall = recall_score(y_t, y_p, average="macro",zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro",zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        print(f"{disease} Results [{backbone}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}")


# ========================
# Predict onsite labels using trained model
# ========================
def predict_onsite_labels(model_path, image_dir, backbone="resnet18", batch_size=32, 
                          img_size=256, output_csv=None, threshold=0.5):
    """
    Predict labels for onsite images using a trained model.
    
    Args:
        model_path: Path to the trained model checkpoint (e.g., "checkpoints/best_resnet18.pt")
        image_dir: Directory containing onsite test images
        backbone: Model backbone ("resnet18" or "efficientnet")
        batch_size: Batch size for inference
        img_size: Image size for preprocessing
        output_csv: Optional path to save predictions as CSV. If None, returns DataFrame.
        threshold: Threshold for binary classification (default: 0.5)
    
    Returns:
        pandas.DataFrame with columns: id, D, G, A
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")
    
    # Create transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset and dataloader
    dataset = RetinaInferenceDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Build and load model
    model = build_model(backbone, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Make predictions
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for imgs, img_files in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)
            
            predictions.extend(preds)
            image_ids.extend(img_files)
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'id': image_ids,
        'D': [pred[0] for pred in predictions],
        'G': [pred[1] for pred in predictions],
        'A': [pred[2] for pred in predictions]
    })
    
    # Sort by image id to ensure consistent ordering
    results_df = results_df.sort_values('id').reset_index(drop=True)
    
    # Save to CSV if output path is provided
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    
    return results_df


def get_args():

    #default values
    task_name = ''
    backbone = 'efficientnet'  # backbone choices: ["resnet18", "efficientnet"]
    train_mode = False
    epochs = 20
    learning_rate = 1e-3
    evaluate_mode = False
    freeze_backbone = False
    loss_mode = 'bce-logits'  # Loss choices: 'bce-logits', 'focal', 'bce-balanced'
    attention_mode = ''  # Attention choices: '', 'se', 'mha'
    
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value.upper()
    print("Parsed arguments:")
    print(args, "\n")
    if(len(args) == 0):
        print("No arguments provided. Avaliable arguments: 'TASK', 'BACKBONE', 'TRAIN', 'EVALUATE', 'LOSS_FUNCTION'")
        print("Example usage: 'python jnp.py TASK=1.2 BACKBONE=RESNET18'")
        sys.exit(1)

    if 'TASK' in args:
        if(args['TASK'] in ('1.1', '1.2', '1.3', '2.1', '2.2', '3.1', '3.2', '4')):
            task_name = args['TASK']
            train_mode, epochs, learning_rate, evaluate_mode, freeze_backbone, loss_mode, attention_mode = get_task_arg(task_name)
        else:
            print("Invalid TASK argument. Supported: '1.1', '1.2', '1.3', '2.1', '2.2', '3.1', '3.2', '4'")
            sys.exit(1)
        if 'BACKBONE' in args:
            if(args['BACKBONE'] in ('RESNET18', 'EFFICIENTNET')):
                backbone = args['BACKBONE'].lower()
            else:
                print("Invalid BACKBONE argument. Supported: 'resnet18', 'efficientnet'")
                sys.exit(1)
        else:
            print("!!! No BACKBONE specified !!! Using default backbone:", backbone)

    else:
        if 'BACKBONE' in args:
            if(args['BACKBONE'] in ('RESNET18', 'EFFICIENTNET')):
                backbone = args['BACKBONE'].lower()
            else:
                print("Invalid BACKBONE argument. Supported: 'resnet18', 'efficientnet'")
                sys.exit(1)

        if 'TRAIN' in args:
            if(args['TRAIN'] in ('1', 'TRUE')):
                train_mode = True
        else:
            train_mode = False

        if 'EPOCHS' in args:
            if args['EPOCHS'].isdigit() and int(args['EPOCHS']) > 0:
                epochs = int(args['EPOCHS'])
            else:
                print("Invalid EPOCHS argument. It should be positive integer.")
                sys.exit(1)
        
        if 'LEARNING_RATE' in args:
            try:
                learning_rate = float(args['LEARNING_RATE'])
                if learning_rate <= 0:
                    raise ValueError
            except ValueError:
                print("Invalid LEARNING_RATE argument. It should be a positive float.")
                sys.exit(1)

        if 'EVALUATE' in args:
            if(args['EVALUATE'] in ('1', 'TRUE')):
                evaluate_mode = True
        else:
            evaluate_mode = False

        if 'FREEZE_BACKBONE' in args:
            if(args['FREEZE_BACKBONE'] in ('1', 'TRUE')):
                freeze_backbone = True
            else:
                freeze_backbone = False

        if 'LOSS_FUNCTION' in args:
            if(args['LOSS_FUNCTION'] in ('FOCAL', 'BCE-LOGITS', 'BCE-BALANCED')):
                loss_mode = args['LOSS_FUNCTION'].lower()
            else:
                print("Invalid LOSS_FUNCTION argument. Supported: 'bce-logits' (default), 'focal', 'bce-balanced'")
                sys.exit(1)
        if 'ATTENTION' in args:
            if(args['ATTENTION'] in ('', 'SE', 'MHA')):
                attention_mode = args['ATTENTION'].lower()
            else:
                print("Invalid ATTENTION argument. Supported: '', 'se', 'mha'")
                sys.exit(1)


    print("Backbone: ", backbone)
    print("Training Mode: ", train_mode)
    print("Epochs: ", epochs)
    print("Learning Rate: ", learning_rate)
    print("Evaluation Mode: ", evaluate_mode)
    print("Loss Mode: ", loss_mode)
    print("Attention Mode: ", attention_mode, "\n")

    return task_name, backbone, train_mode, epochs, learning_rate, evaluate_mode, freeze_backbone, loss_mode, attention_mode



def get_task_arg(task_name):

    print("########################")
    print(f"Task: {task_name}")
    print("########################")

    # Structure:
    # train, epochs, learning_rate, evaluate_mode, freeze_backbone, loss_mode, attention_mode
    if task_name == '1.1':
        return True, 0, 0, True, False, 'bce-logits', ''
    elif task_name == '1.2':
        return True, 20, 1e-3, True, True, 'bce-logits', ''
    elif task_name == '1.3':
        return True, 20, 1e-3, True, False, 'bce-logits', ''
    elif task_name == '2.1':
        return True, 20, 1e-4, True, False, 'focal', ''
    elif task_name == '2.2':
        return True, 20, 1e-4, True, False, 'bce-balanced', ''
    elif task_name == '3.1':
        return 1
    elif task_name == '3.2':
        return 1
    elif task_name == '4':
        return 1
    
    return 1


# ========================
# main
# ========================
if __name__ == "__main__":

    task_name, backbone, train_mode, epochs, learning_rate, evaluate_mode, freeze_backbone, loss_mode, attention_mode = get_args()
            
    if train_mode:
        train_csv = "train.csv" # replace with your own train label file path
        val_csv   = "val.csv" # replace with your own validation label file path
        test_csv  = "offsite_test.csv"  # replace with your own test label file path
        train_image_dir ="./images/train"   # replace with your own train image floder path
        val_image_dir = "./images/val"  # replace with your own validation image floder path
        test_image_dir = "./images/offsite_test" # replace with your own test image floder path
        if backbone == 'resnet18':
            pretrained_backbone = './pretrained_backbone/ckpt_resnet18_ep50.pt'  # replace with your own pretrained backbone path
        elif backbone == 'efficientnet':
            pretrained_backbone = './pretrained_backbone/ckpt_efficientnet_ep50.pt'  # replace with your own pretrained backbone path
        #backbone = 'efficientnet'  # backbone choices: ["resnet18", "efficientnet"]
        #freeze_backbone = False  # Set to True to freeze backbone during training
        #loss = 'bce-logits'  # Loss choices: 'focal', 'bce-logits', 'bce-balanced'
        
        train_one_backbone(
            backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
            epochs=epochs, batch_size=32, lr=learning_rate, img_size=256, pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone, loss=loss_mode
        )


    if evaluate_mode:
        test_image_dir = "./images/onsite_test"
        #backbone = "efficientnet"
        model_path = f"./checkpoints/best_{backbone}.pt"
        batch_size = 32
        img_size = 256
        if task_name == '':
            output_csv = f"./all_results/onsite_predictions_{backbone}.csv"
        else:
            output_csv = f"./task_results/onsite_predictions_{backbone}_task_{task_name}.csv"

        print("\n" + "="*50)
        print("Predicting onsite labels...")
        print("="*50)
        onsite_predictions = predict_onsite_labels(
            model_path=model_path,
            image_dir=test_image_dir,
            backbone=backbone,
            batch_size=batch_size,
            img_size=img_size,
            output_csv=output_csv
        )
        print(f"\nPredicted labels for {len(onsite_predictions)} onsite images")
        print("\nFirst few predictions:")
        print(onsite_predictions.head())
