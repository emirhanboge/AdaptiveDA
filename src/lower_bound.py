from dataset_creation import create_datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50

def train_and_evaluate():
    aid_dataset_path = "data/AID"
    ucm_dataset_path = "data/UCMerced_LandUse/Images"

    aid_dataset, ucm_dataset = create_datasets(aid_dataset_path, ucm_dataset_path)

    batch_size = 64
    num_classes = len(aid_dataset.common_classes)
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aid_train_loader = DataLoader(aid_dataset, batch_size=batch_size, shuffle=True)
    ucm_test_loader = DataLoader(ucm_dataset, batch_size=batch_size, shuffle=False)

    print("Training AID -> UCM")
    _train_and_evaluate_model(aid_train_loader, ucm_test_loader, num_classes, num_epochs, device)

    ucm_train_loader = DataLoader(ucm_dataset, batch_size=batch_size, shuffle=True)
    aid_test_loader = DataLoader(aid_dataset, batch_size=batch_size, shuffle=False)

    print("Training UCM -> AID")
    _train_and_evaluate_model(ucm_train_loader, aid_test_loader, num_classes, num_epochs, device)


def _train_and_evaluate_model(train_loader, test_loader, num_classes, num_epochs, device):
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on target domain: {accuracy}%")

if __name__ == '__main__':
    train_and_evaluate()

