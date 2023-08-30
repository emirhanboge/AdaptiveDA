from utils import ResNet50_DA, coral, setup_seed
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch

def train_and_test_domain_adaptation(train_loader_source, train_loader_target, test_loader, num_classes, adaptive_dropout, adaptive_bn, lambda_coral=0.5, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50_DA(num_classes=num_classes, adaptive_dropout=adaptive_dropout, adaptive_bn=adaptive_bn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        iter_source = iter(train_loader_source)
        iter_target = iter(train_loader_target)
        num_batches = min(len(iter_source), len(iter_target))
        running_loss = 0.0

        for _ in tqdm(range(num_batches), total=num_batches):
            optimizer.zero_grad()
            try:
                target_inputs, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(train_loader_target)
                target_inputs, _ = next(iter_target)

            source_inputs, source_labels = next(iter_source)
            inputs, source_labels = source_inputs.to(device), source_labels.to(device)
            class_output_source = model(inputs, domain_mask=0)

            inputs = target_inputs.to(device)
            class_output_target = model(inputs, domain_mask=1)

            class_loss = criterion(class_output_source, source_labels)
            coral_loss = coral(class_output_source, class_output_target)

            loss = class_loss + lambda_coral * coral_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            class_outputs = model(inputs)
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on target domain: {100 * correct / total}%")


if __name__ == "__main__":
    setup_seed(42)
    aid_path = "data/AID"
    ucm_path = "data/UCM"
    aid_dataset, ucm_dataset = create_datasets(aid_path, ucm_path)
    aid_train_loader = DataLoader(aid_dataset, batch_size=64, shuffle=True, drop_last=True)
    ucm_train_loader = DataLoader(ucm_dataset, batch_size=64, shuffle=True, drop_last=True)
    ucm_test_loader = DataLoader(ucm_dataset, batch_size=64, shuffle=False, drop_last=True)
    aid_test_loader = DataLoader(aid_dataset, batch_size=64, shuffle=False, drop_last=True)

    common_classes = aid_dataset.common_classes

    print("Training AID -> UCM Adaptive Dropout and Adaptive BN")
    train_and_test_domain_adaptation(aid_train_loader, ucm_train_loader, ucm_test_loader, len(common_classes), True, True)

    print("Training UCM -> AID Adaptive Dropout and Adaptive BN")
    train_and_test_domain_adaptation(ucm_train_loader, aid_train_loader, aid_test_loader, len(common_classes), True, True)

    print("Training AID -> UCM Standard Dropout and Adaptive BN")
    train_and_test_domain_adaptation(aid_train_loader, ucm_train_loader, ucm_test_loader, len(common_classes), False, True)

    print("Training UCM -> AID Standard Dropout and Adaptive BN")
    train_and_test_domain_adaptation(ucm_train_loader, aid_train_loader, aid_test_loader, len(common_classes), False, True)

    print("Training AID -> UCM Adaptive Dropout and Standard BN")
    train_and_test_domain_adaptation(aid_train_loader, ucm_train_loader, ucm_test_loader, len(common_classes), True, False)

    print("Training UCM -> AID Adaptive Dropout and Standard BN")
    train_and_test_domain_adaptation(ucm_train_loader, aid_train_loader, aid_test_loader, len(common_classes), True, False)

    print("Training AID -> UCM Standard Dropout and Standard BN")
    train_and_test_domain_adaptation(aid_train_loader, ucm_train_loader, ucm_test_loader, len(common_classes), False, False)

    print("Training UCM -> AID Standard Dropout and Standard BN")
    train_and_test_domain_adaptation(ucm_train_loader, aid_train_loader, aid_test_loader, len(common_classes), False, False)
