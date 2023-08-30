from utils import setup_seed, ResNet50_DA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train_model(aid_train_loader, ucm_train_loader, common_classes, adaptive_dropout, adaptive_bn, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50_DA(num_classes=len(common_classes), adaptive_dropout=adaptive_dropout, adaptive_bn=adaptive_bn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        aid_iter = iter(aid_train_loader)
        ucm_iter = iter(ucm_train_loader)
        num_batches = min(len(aid_iter), len(ucm_iter))

        running_loss = 0.0
        i = 0
        for _ in tqdm(range(num_batches), total=num_batches):
            optimizer.zero_grad()
            p = float(i + epoch * num_batches) / num_epochs / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            try:
                ucm_inputs, _ = next(ucm_iter)
            except:
                ucm_iter = iter(ucm_train_loader)
                ucm_inputs, _ = next(ucm_iter)

            aid_inputs, aid_labels = next(aid_iter)
            aid_domains = torch.zeros(len(aid_inputs), dtype=torch.long)
            ucm_domains = torch.ones(len(ucm_inputs), dtype=torch.long)

            inputs, aid_labels, aid_domains = aid_inputs.to(device), aid_labels.to(device), aid_domains.to(device)
            class_outputs, s_domain_output = model(inputs, alpha)

            inputs, ucm_domains = ucm_inputs.to(device), ucm_domains.to(device)
            _, t_domain_output = model(inputs, alpha)

            class_loss = criterion(class_outputs, aid_labels)
            s_domain_loss = criterion(s_domain_output, aid_domains)
            t_domain_loss = criterion(t_domain_output, ucm_domains)
            loss = s_domain_loss + class_loss + t_domain_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            i += 1

    return model

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)
            class_outputs, _ = model(inputs, alpha=0)
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on target domain: {100 * correct / total}%")


if __name__ == "__main__":
    aid_dataset_path = "data/AID"
    ucm_dataset_path = "data/UCM"
    aid_dataset, ucm_dataset = create_datasets(aid_dataset_path, ucm_dataset_path)

    common_classes = aid_dataset.common_classes

    batch_size = 64
    aid_train_loader = DataLoader(aid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    ucm_test_loader = DataLoader(ucm_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    ucm_train_loader = DataLoader(ucm_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    aid_test_loader = DataLoader(aid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    setup_seed(42)

    print("Training AID -> UCM with Adaptive Dropout and Adaptive BN")
    model = train_model(aid_train_loader, ucm_train_loader, common_classes, True, True)
    test_model(model, ucm_test_loader)

    print("Training UCM -> AID with Adaptive Dropout and Adaptive BN")
    model = train_model(ucm_train_loader, aid_train_loader, common_classes, True, True)
    test_model(model, aid_test_loader)

    print("Training AID -> UCM with Standard Dropout and Adaptive BN")
    model = train_model(aid_train_loader, ucm_train_loader, common_classes, False, True)
    test_model(model, ucm_test_loader)

    print("Training UCM -> AID with Standard Dropout and Adaptive BN")
    model = train_model(ucm_train_loader, aid_train_loader, common_classes, False, True)
    test_model(model, aid_test_loader)

    print("Training AID -> UCM with Adaptive Dropout and Standard BN")
    model = train_model(aid_train_loader, ucm_train_loader, common_classes, True, False)
    test_model(model, ucm_test_loader)

    print("Training UCM -> AID with Adaptive Dropout and Standard BN")
    model = train_model(ucm_train_loader, aid_train_loader, common_classes, True, False)
    test_model(model, aid_test_loader)

    print("Training AID -> UCM with Standard Dropout and Standard BN")
    model = train_model(aid_train_loader, ucm_train_loader, common_classes, False, False)
    test_model(model, ucm_test_loader)

    print("Training UCM -> AID with Standard Dropout and Standard BN")
    model = train_model(ucm_train_loader, aid_train_loader, common_classes, False, False)
    test_model(model, aid_test_loader)

