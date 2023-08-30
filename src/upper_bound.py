from dataset_creation import create_datasets
from utils import train_test_split_by_class
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

class SubsetDataset(CustomDataset):
    def __init__(self, data, subset_data, transform=None):
        super().__init__(data.path, data.common_classes, data.dataset_type, transform)
        self.subset_data = subset_data

    def __len__(self):
        return len(self.subset_data)

    def __getitem__(self, idx):
        img_path, label = self.subset_data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def upper_bound_train_and_evaluate():
    aid_dataset_path = "data/AID"
    ucm_dataset_path = "data/UCMerced_LandUse/Images"

    aid_dataset, ucm_dataset = create_datasets(aid_dataset_path, ucm_dataset_path)

    input_size = 256

    data_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ucm_train_data, ucm_test_data = train_test_split_by_class(ucm_dataset, test_size=0.8, random_state=42)
    aid_train_data, aid_test_data = train_test_split_by_class(aid_dataset, test_size=0.8, random_state=42)

    batch_size = 64
    aid_train_loader = DataLoader(aid_dataset, batch_size=batch_size, shuffle=True)
    ucm_train_loader = DataLoader(SubsetDataset(ucm_dataset, ucm_train_data, data_transform), batch_size=batch_size, shuffle=True)
    ucm_test_loader = DataLoader(SubsetDataset(ucm_dataset, ucm_test_data, data_transform), batch_size=batch_size, shuffle=False)
    aid_test_loader = DataLoader(SubsetDataset(aid_dataset, aid_test_data, data_transform), batch_size=batch_size, shuffle=False)

    num_epochs = 20
    num_classes = len(aid_dataset.common_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training AID + 20% UCM -> 80% UCM")
    _train_evaluate(aid_train_loader, ucm_train_loader, ucm_test_loader, num_classes, num_epochs, device)

    print("Training UCM + 20% AID -> 80% AID")
    _train_evaluate(ucm_train_loader, aid_train_loader, aid_test_loader, num_classes, num_epochs, device)

def _train_evaluate(train_loader1, train_loader2, test_loader, num_classes, num_epochs, device):
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(train_loader1), total=len(train_loader1)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader1)}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(train_loader2), total=len(train_loader2)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader2)}")

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
    print(f"Accuracy on target domain: {100 * correct / total}%")

if __name__ == '__main__':
    upper_bound_train_and_evaluate()

