from dataloader_style import WikiArtDataset
from transformers import AutoImageProcessor, AutoModel
import torch
from torch.utils.data import DataLoader
from dino_style import DINOStyle
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


def train(model, train_loader, test_loader):

    # wandb
    wandb.init(project="DINOcls", entity="bilardellodavides-vmart", name="dino-style-training-CLS")

    # Params
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-4
    weight_decay = 1e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()


    for e in tqdm(range(epochs)):
        print(f"Epoch {e+1}/{epochs}")

        # metti il model in train
        model.train()

        all_preds = []
        all_labels = []

        total_loss = 0.0
        for data, style_gt in train_loader:

            # sposto i dati sul device
            data = data.to(device)
            style_gt = style_gt.to(device)

            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, style_gt)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(style_gt.cpu().numpy())

        # Calcola metriche
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0)

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {e+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        wandb.log({
            "train_loss": avg_loss,
            "train_accuracy": acc,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1_score": f1
        })

        # valiazione
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, style_gt in test_loader:
                data = data.to(device)
                style_gt = style_gt.to(device)

                outputs = model(data)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(style_gt.cpu().numpy())
        
        # Calcola metriche di validazione
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
        print(f"Validation - Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        wandb.log({
            "val_accuracy": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1,
        })
        

if __name__ == '__main__':
    DATASET_PATH = '/work/cvcs2025/bilardello_melis_prato/wikiart'
    CSV_PATH = '/work/cvcs2025/bilardello_melis_prato/wikiart/wclasses.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocessing per dinov2
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small',
                                                    use_fast=True)

    # carico il dataset
    dataset = WikiArtDataset(csv_file=CSV_PATH, root_dir=DATASET_PATH, transform=processor)

    # splitto il dataset in train e test
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_set, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)

    model = DINOStyle(in_channels=384, num_classes=len(dataset.styles)).to(device)

    train(model, train_loader, test_loader)

