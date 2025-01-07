import torch
from pathlib import Path
import numpy as np
import logging
from src.logging_config.logging_setup import setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

setup_logging()
logger = logging.getLogger(__name__)


def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float()
    return correct.sum() / len(correct)


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    for data, labels, _ in train_loader:  # Ignore patient_id
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += calculate_accuracy(outputs, labels).item()
    return train_loss / len(train_loader), train_acc / len(train_loader)


def validate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data, labels, _ in test_loader:  # Ignore patient_id
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += calculate_accuracy(outputs, labels).item()

            all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())

    return (
        test_loss / len(test_loader),
        test_acc / len(test_loader),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )
