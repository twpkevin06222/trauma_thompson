import torch
import argparse
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from utils import get_model, VideoDataset, get_video_transform
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    required=True,
    help="path to the dataset directory",
)
parser.add_argument("--save_dir", type=str, default="./outputs")
parser.add_argument("--num_classes", type=int, default=124)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--epoch_chck_point", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--test_only", action="store_true", help="Test minimal run")
parser.add_argument(
    "--resume",
    action="store_true",
    help="resume training from checkpoint stored in the 'save_dir'",
)
parser.add_argument(
    "--experiment_name", type=str, default="exp_01", help="experiment name for mlflow"
)
parser.add_argument(
    "--model_verbosity", action="store_true", help="enable model verbosity"
)
args = parser.parse_args()
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(f"task1_{args.experiment_name}")
MODEL_NAME = "facebook/vjepa2-vitl-fpc16-256-ssv2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create save_dir if it doesn't exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


def run():
    with mlflow.start_run():
        params = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "num_classes": args.num_classes,
            "save_dir": args.save_dir,
            "backbone": MODEL_NAME,
        }
        mlflow.log_params(params)
        csv_pth = "./outputs/data_prep_split.csv"
        video_dir = args.dataset_dir
        model, processor = get_model(
            model_name=MODEL_NAME,
            num_classes=args.num_classes,
            verbosity=args.model_verbosity,
            device=device,
        )
        video_transform = get_video_transform(prob=0.25)
        val_dataset = VideoDataset(csv_pth, video_dir, mode="val")
        train_dataset = VideoDataset(
            csv_pth, video_dir, transform=video_transform, mode="train"
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        # only the head will be trained
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
        best_acc = 0.0
        if args.test_only:
            print("-" * 100)
            print(
                "This run is on TEST mode, please check if this is the correct configuration ..."
            )
            print("-" * 100)
        if args.resume:
            print("-" * 100)
            print(
                "This run is on RESUME mode, please check if this is the correct configuration ..."
            )
            resume_pth = os.path.join(args.save_dir, "last.pth")
            print(f"Resuming from {resume_pth} ...")
            checkpoint = torch.load(resume_pth)
            model.classifier.load_state_dict(checkpoint["head"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            print("-" * 100)
        else:
            start_epoch = 0
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            print("Training ...")
            for step, (batch_video, label) in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):
                batch_size = batch_video.shape[0]
                label = label.to(device)
                # Use processor to resize/crop frames to the model's expected size
                inputs = processor(batch_video, return_tensors="pt").to(device)
                if batch_size == 1:
                    inputs = inputs.pixel_values_videos.squeeze()
                    inputs = inputs.unsqueeze(0)
                else:
                    inputs = inputs.pixel_values_videos.squeeze()
                outputs = model(inputs, labels=label)
                loss = outputs.loss
                # backpropagate the loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                logits = model(inputs).logits
                preds = logits.argmax(-1)
                train_correct += (preds == label).sum().item()
                train_total += label.shape[0]
                if args.test_only:
                    break
            train_acc = train_correct / train_total
            print(
                f"Epoch {epoch + 1}, Train Loss {running_loss}, Train Accuracy {train_acc}"
            )
            mlflow.log_metric("train_loss", running_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            # save checkpoint
            if epoch % args.epoch_chck_point == 0:
                print("Saving checkpoint ...")
                checkpoint = {
                    "head": model.classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint, os.path.join(args.save_dir, "last.pth"))
            print("Processing evaluation ...")
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            with torch.no_grad():
                for batch_video, label in tqdm(val_loader, total=len(val_loader)):
                    batch_size = batch_video.shape[0]
                    label = label.to(device)
                    # Use processor to resize/crop frames to the model's expected size
                    inputs = processor(batch_video, return_tensors="pt").to(device)
                    if batch_size == 1:
                        inputs = inputs.pixel_values_videos.squeeze()
                        inputs = inputs.unsqueeze(0)
                    else:
                        inputs = inputs.pixel_values_videos.squeeze()
                    loss = model(inputs, labels=label).loss
                    logits = model(inputs).logits
                    preds = logits.argmax(-1)
                    val_correct += (preds == label).sum().item()
                    val_total += label.shape[0]
                    val_loss += loss.item()
                    if args.test_only:
                        break
            val_acc = val_correct / val_total
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
            print(
                f"Epoch {epoch + 1}, Validation Loss {val_loss}, Validation Accuracy {val_acc}"
            )
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print()


if __name__ == "__main__":
    run()
