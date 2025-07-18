import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

import wandb


class VMARTResnet(nn.Module):

    def __init__(self, config, device, wandb, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.wandb = wandb
        self.config = config
        self.finetune = config["finetune"]

        self.device = device

        self.model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)

        # Input del layer FC
        self.fc_input_dim = self.model.classifier[1].in_features

        # Togliamo il layer FC della resnet
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for param in self.model.parameters():
            param.requires_grad = self.finetune

        self.hidden_dim = 512

        self.style_head_out_dim = 27  # style head dim
        self.genre_head_out_dim = 10  # genre head dim

        # use 3 hidden layer with dropout
        self.style_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.fc_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.style_head_out_dim),
        )

        self.genre_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.fc_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.genre_head_out_dim),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze(-1).squeeze(-1)  # (N, 512, 1, 1) -> (N, 512))
        return self.style_head(x), self.genre_head(x)

    def load_weights(self, style_weights_path, genre_weights_path, model_weights_path):
        # pesi dello style
        if os.path.exists(style_weights_path):
            self.style_head.load_state_dict(torch.load(style_weights_path))
            print(f"Loaded style head weights from {style_weights_path}")
        else:
            print(f"Style head weights not found at {style_weights_path}")

        # pesi del genere
        if os.path.exists(genre_weights_path):
            self.genre_head.load_state_dict(torch.load(genre_weights_path))
            print(f"Loaded genre head weights from {genre_weights_path}")
        else:
            print(f"Genre head weights not found at {genre_weights_path}")

        # pesi del modello
        if os.path.exists(model_weights_path):
            full_state_dict = torch.load(model_weights_path)
            model_state_dict = {}
            for key, value in full_state_dict.items():
                if key.startswith('model.'):
                    model_state_dict[key[6:]] = value
            self.model.load_state_dict(model_state_dict)
            print(f"Loaded model weights from {model_weights_path}")
        else:
            print(f"Model weights not found at {model_weights_path}")

    def predict(self, x):
        with torch.no_grad():
            style, genre = self.forward(x)
            print(torch.softmax(style, dim=1).max() * 100, torch.softmax(genre, dim=1).max() * 100)

            return torch.argmax(style), torch.argmax(genre)

    def save_weights(self, subfolder, optimizer=None):
        os.makedirs(f"weights/{subfolder}", exist_ok=True)
        style_path = f"weights/{subfolder}/vmart_resnet_weights_style_head.pth"
        genre_path = f"weights/{subfolder}/vmart_resnet_weights_genre_head.pth"
        model_path = ""
        if self.finetune:
            model_path = f"weights/{subfolder}/vmart_resnet_weights_model.pth"

        torch.save(self.style_head.state_dict(), style_path)
        torch.save(self.genre_head.state_dict(), genre_path)
        if self.finetune:
            torch.save(self.state_dict(), model_path)

        if optimizer:
            optimizer_path = f"weights/{subfolder}/optimizer.pth"
            torch.save(optimizer.state_dict(), optimizer_path)

        if self.wandb is not None:
            wandb.save(style_path)
            wandb.save(genre_path)
            if self.finetune:
                wandb.save(model_path)
            if optimizer:
                wandb.save(f"weights/{subfolder}/optimizer.pth")

    def trainResNet(self, train_loader, test_loader):
        epoch = self.config['epochs']
        patience = self.config['patience']
        n_epoch_save = self.config['n_epochs_save']
        save_weights = self.config['save_weights']
        weight_decay = self.config['weight_decay']
        lr_backbone = self.config['learning_rate_model']
        lr_head = self.config['learning_rate_head']
        finetune = self.config['finetune']

        if finetune:
            optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': lr_backbone},
                {'params': self.style_head.parameters(), 'lr': lr_head},
                {'params': self.genre_head.parameters(), 'lr': lr_head}
            ], weight_decay=weight_decay)
        else:
            optimizer = optim.Adam([
                {'params': self.style_head.parameters(), 'lr': lr_head},
                {'params': self.genre_head.parameters(), 'lr': lr_head}
            ], weight_decay=weight_decay)

        if self.config["resume"]["checkpoint"]:
            optimizer_path = self.config["resume"]["optimizer"]
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path))
                print(f"Loaded optimizer state from {optimizer_path}")
            else:
                print(f"Optimizer state not found at {optimizer_path}")

        criterion = nn.CrossEntropyLoss()

        scheduler = None
        if finetune:
            warmup_epochs = self.config['warmup_epochs_for_finetune']
            num_train_steps_per_epoch = len(train_loader)

            warmup_steps = warmup_epochs * num_train_steps_per_epoch
            total_steps = epoch * num_train_steps_per_epoch

            # Numero di step a LR zero per le head
            head_zero_steps = self.config.get('head_zero_steps', 5) * num_train_steps_per_epoch

            # Scheduler 0: LR zero iniziale per le head, backbone invariata
            zero_lambdas = [
                lambda step: 1.0,  # backbone
                lambda step: 0.0 if step < head_zero_steps else 1.0,  # style head
                lambda step: 0.0 if step < head_zero_steps else 1.0  # genre head
            ]
            zero_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=zero_lambdas)

            # Scheduler 1: Warmup lineare per le head, backbone lr costante
            start_factor = self.config.get('warmup_start_factor', 0.1)
            warmup_lambdas = [
                lambda step: 1.0,
                lambda step, sf=start_factor, ws=warmup_steps: sf + (1 - sf) * ((step + 1) / ws) if step < ws else 1.0,
                lambda step, sf=start_factor, ws=warmup_steps: sf + (1 - sf) * ((step + 1) / ws) if step < ws else 1.0
            ]
            warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambdas)

            # Scheduler 2: Decay Cosine per le head, backbone lr costante
            eta_min = self.config.get('eta_min', 1e-6)
            decay_steps = warmup_epochs * num_train_steps_per_epoch
            main_lambdas = [
                lambda step: 1.0,
                lambda step, ds=decay_steps, em=eta_min: em + (1 - em) * 0.5 * (
                        1 + math.cos(math.pi * ((step % ds) + 1) / ds)),
                lambda step, ds=decay_steps, em=eta_min: em + (1 - em) * 0.5 * (
                        1 + math.cos(math.pi * ((step % ds) + 1) / ds))
            ]
            main_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=main_lambdas)

            # Combina i tre scheduler in sequenza: zero, warmup, decay per le head
            scheduler = lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[zero_scheduler, warmup_scheduler, main_scheduler],
                milestones=[head_zero_steps, head_zero_steps + warmup_steps]
            )

        if self.config["resume"]["checkpoint"]:
            top_loss = self.config["resume"]["top_vLoss"]
        else:
            top_loss = np.inf

        count_patience = 0

        for e in range(epoch):

            self.train()

            tloss = 0
            ee = 0
            for data, style_gt, genre_gt in train_loader:
                style_gt = style_gt.to(self.device)
                genre_gt = genre_gt.to(self.device)
                data = data.to(self.device)
                optimizer.zero_grad()

                style_pred, genre_pred = self(data)

                if finetune:
                    loss_style = criterion(style_pred, style_gt)
                    loss_genre = criterion(genre_pred, genre_gt)

                    loss = (0.5 * loss_style) + (0.5 * loss_genre)
                    loss.backward()
                    optimizer.step()

                    if scheduler:
                        scheduler.step()

                    tloss += loss.item()
                else:
                    loss_style = criterion(style_pred, style_gt)
                    loss_genre = criterion(genre_pred, genre_gt)

                    loss_style.backward()
                    loss_genre.backward()

                    optimizer.step()

                    loss = (0.5 * loss_style) + (0.5 * loss_genre)
                    tloss += loss.item()

                ee += 1
            tloss /= len(train_loader)
            print(f"Epoch {e + 1}/{epoch} - Train Loss: {tloss:.4f}")

            vLoss, style_accuracy, genre_accuracy, style_report, genre_report, style_cm, genre_cm = self.validateResNet(
                test_loader)

            # Log metriche base
            log_dict = {
                "train_loss": tloss,
                "val_loss": vLoss,
                "style_accuracy": style_accuracy,
                "style_precision": style_report['weighted avg']['precision'],
                "style_recall": style_report['weighted avg']['recall'],
                "genre_accuracy": genre_accuracy,
                "genre_precision": genre_report['weighted avg']['precision'],
                "genre_recall": genre_report['weighted avg']['recall'],
            }

            if finetune and self.wandb is not None:
                log_dict["lr_backbone"] = optimizer.param_groups[0]['lr']
                log_dict["lr_head"] = optimizer.param_groups[1]['lr']

            # Creo e invio la confusion matrix come immagine
            try:

                matplotlib.use('Agg')  # Uso il backend Agg che non richiede un display

                # Crea directory per salvare le immagini delle confusion matrix
                os.makedirs("confusion_matrices", exist_ok=True)
                style_cm_path = f"confusion_matrices/style_cm_epoch_{e + 1}.png"
                genre_cm_path = f"confusion_matrices/genre_cm_epoch_{e + 1}.png"

                # Confusion Matrix per stile
                fig_style, ax_style = plt.subplots(figsize=(10, 10))
                im_style = ax_style.imshow(style_cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax_style.set_title("Style Confusion Matrix")
                fig_style.colorbar(im_style, ax=ax_style)
                # Salva la figura localmente
                fig_style.savefig(style_cm_path)
                # Invia a wandb sia come immagine che come file
                if self.wandb is not None:
                    log_dict["style_confusion_matrix"] = wandb.Image(fig_style)
                    wandb.save(style_cm_path)
                plt.close(fig_style)

                # Confusion Matrix per genere
                fig_genre, ax_genre = plt.subplots(figsize=(10, 10))
                im_genre = ax_genre.imshow(genre_cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax_genre.set_title("Genre Confusion Matrix")
                fig_genre.colorbar(im_genre, ax=ax_genre)
                # Salva la figura localmente
                fig_genre.savefig(genre_cm_path)
                # Invia a wandb sia come immagine che come file
                if self.wandb is not None:
                    log_dict["genre_confusion_matrix"] = wandb.Image(fig_genre)
                    self.wandb.save(genre_cm_path)
                plt.close(fig_genre)

                print(f"Confusion matrices saved to {style_cm_path} and {genre_cm_path}")
            except Exception as e:
                print(f"Error creating confusion matrix plots: {e}")

            # Invia tutto a wandb
            if self.wandb is not None:
                self.wandb.log(log_dict)

            if vLoss < top_loss:
                top_loss = vLoss

                if save_weights:
                    self.save_weights("best", optimizer)

                count_patience = 0
            else:
                count_patience += 1

            if (e + 1) % n_epoch_save == 0:
                if save_weights:
                    self.save_weights(f"{e + 1}", optimizer)

            if count_patience == patience:
                break

        if save_weights:
            self.save_weights(f"last", optimizer)

    def validateResNet(self, test_loader):

        self.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct_style = 0
        correct_genre = 0
        total_samples = 0

        # Raccolgo tutte le predizioni e i ground truth per calcolare la confusion matrix
        all_style_gt = []
        all_style_pred = []
        all_genre_gt = []
        all_genre_pred = []

        with torch.no_grad():
            for data, style_gt, genre_gt in test_loader:
                style_gt = style_gt.to(self.device)
                genre_gt = genre_gt.to(self.device)
                data = data.to(self.device)

                style_pred, genre_pred = self(data)

                loss_style = criterion(style_pred, style_gt)
                loss_genre = criterion(genre_pred, genre_gt)

                loss = (0.5 * loss_style) + (0.5 * loss_genre)
                total_loss += loss.item()

                # predizioni per ogni immagine del batch
                _, predicted_style = torch.max(style_pred.data, 1)  # (N, 27)
                _, predicted_genre = torch.max(genre_pred.data, 1)

                total_samples += style_gt.size(0)
                correct_style += (predicted_style == style_gt).sum().item()
                correct_genre += (predicted_genre == genre_gt).sum().item()

                # Salvo le predizioni e i ground truth per la confusion matrix
                all_style_gt.extend(style_gt.cpu().numpy())
                all_style_pred.extend(predicted_style.cpu().numpy())
                all_genre_gt.extend(genre_gt.cpu().numpy())
                all_genre_pred.extend(predicted_genre.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        style_accuracy = 100 * correct_style / total_samples
        genre_accuracy = 100 * correct_genre / total_samples

        # Calcolo precision e recall per stile e genere
        style_report = classification_report(all_style_gt, all_style_pred, output_dict=True, zero_division=0)
        genre_report = classification_report(all_genre_gt, all_genre_pred, output_dict=True, zero_division=0)

        # Calcolo confusion matrix per stile e genere
        style_cm = confusion_matrix(all_style_gt, all_style_pred)
        genre_cm = confusion_matrix(all_genre_gt, all_genre_pred)

        print(f'Validation Loss: {avg_loss:.4f}')
        print(f'Style Accuracy: {style_accuracy:.2f}%')
        print(f'Genre Accuracy: {genre_accuracy:.2f}%')

        # Stampo precision e recall
        print(f"Style Precision (Weighted): {style_report['weighted avg']['precision']:.4f}")
        print(f"Style Recall (Weighted): {style_report['weighted avg']['recall']:.4f}")
        print(f"Genre Precision (Weighted): {genre_report['weighted avg']['precision']:.4f}")
        print(f"Genre Recall (Weighted): {genre_report['weighted avg']['recall']:.4f}")

        return avg_loss, style_accuracy, genre_accuracy, style_report, genre_report, style_cm, genre_cm
