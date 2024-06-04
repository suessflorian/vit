import argparse
import torch
import torchvision
import checkpoint
import data
import os
import csv
import upload

# NOTE: only nessecary for CUDA devices
from torch.cuda.amp import GradScaler, autocast
from typing import Tuple, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Exploring CIFAR-100 S-NN models")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train for")
parser.add_argument("--batch", type=int, default=128, help="input batch size for training")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--device", type=str, default="cpu", help="device to lay tensor work over")
parser.add_argument("--gcs", action="store_true", help="if model checkpoints should be pushed to gcs")

CIFAR_100_CLASSES = 100
MODEL_NAME = "vit_b_16"

def evaluate(
    model: torch.nn.Module,
    criterion: Callable,
    loader: DataLoader,
    device: str = "cpu",
    scaler: GradScaler | None = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0, 0, 0

    (images, labels) = next(iter(loader))
    batch = images.shape[0]
    with torch.no_grad():
        with tqdm(loader, desc="Evaluation", unit="image", unit_scale=batch) as progress:
            for images, labels in progress:
                images, labels = images.to(device), labels.to(device)
                if scaler:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                progress.set_postfix(test_accuracy=(correct/total))

    average_loss = total_loss / len(loader)
    accuracy = correct / total
    return average_loss, accuracy

def main():
    args = parser.parse_args()

    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    # honed for classes of the CIFAR100 dataset
    model.heads = torch.nn.Sequential(torch.nn.Linear(model.hidden_dim, CIFAR_100_CLASSES))

    preprocess = weights.transforms()

    loaded, loadedModel, metadata = checkpoint.load(model, name=MODEL_NAME, gcs=args.gcs)
    if loaded:
        model = loadedModel


    freeze = [
            "conv_proj",
            "encoder.pos_embedding",
            "encoder.layers.encoder_layer_0.",
            "encoder.layers.encoder_layer_1.",
            "encoder.layers.encoder_layer_2.",
    ]

    for name, param in model.named_parameters():
        for prefix in freeze:
            if any(name.startswith(layer) for layer in prefix):
                param.requires_grad = False

    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    train_loader, test_loader = data.cifar100(preprocess, args.batch, args.device)

    # NOTE: TRAINING
    model.train()
    best_accuracy, best_loss = metadata.accuracy, metadata.loss


    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None


    report = f"./results/train/{MODEL_NAME}.csv"
    report_gcs = f"results/train/{MODEL_NAME}.csv"

    def write_to_csv(path, data):
        if not os.path.exists(path):
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with open(path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "loss", "accuracy"])

        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    best_loss, best_accuracy = evaluate(model, criterion, test_loader, args.device, scaler)
    if metadata.epoch == 0:
        write_to_csv(report, [0, best_loss, best_accuracy])
        if args.gcs:
            upload.gcs(report, report_gcs)


    for i in range(1, args.epochs + 1):
        correct, total = 0, 0
        with tqdm(train_loader, unit="images", unit_scale=args.batch) as progress:
            for images, labels in progress:
                progress.set_description(f"Epoch {i + metadata.epoch}")
                images, labels = images.to(args.device), labels.to(args.device)

                if scaler:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                optimizer.zero_grad(set_to_none=True)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                progress.set_postfix(train_accuracy=f"{(correct/total):.2f}")

        loss, accuracy = evaluate(model, criterion, test_loader, args.device, scaler)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_loss = loss
            checkpoint.cache(
                model,
                checkpoint.Metadata(
                    name=MODEL_NAME, epoch=i + metadata.epoch, accuracy=best_accuracy, loss=best_loss
                ),
                gcs=args.gcs,
            )
        write_to_csv(report, [i + metadata.epoch, loss, accuracy])
        if args.gcs:
            upload.gcs(report, report_gcs)

if __name__ == '__main__':
    main()
