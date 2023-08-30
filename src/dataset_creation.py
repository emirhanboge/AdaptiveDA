import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class RSDataset(Dataset):
    def __init__(self, dataset_path, common_classes, dataset_type, transform=None):
        self.path = dataset_path
        self.common_classes = common_classes
        self.dataset_type = dataset_type
        self.transform = transform
        self.data = []

        for label, (aid_class, ucm_class) in enumerate(common_classes):
            class_name = aid_class if dataset_type == 'AID' else ucm_class
            class_path = os.path.join(self.path, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def create_datasets(aid_dataset_path, ucm_dataset_path):
    input_size = 256

    data_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    common_classes = [
        ('Farmland', 'agricultural'),
        ('Airport', 'airplane'),
        ('BaseballField', 'baseballdiamond'),
        ('Beach', 'beach'),
        ('DenseResidential', 'denseresidential'),
        ('Forest', 'forest'),
        ('MediumResidential', 'mediumresidential'),
        ('Parking', 'parkinglot'),
        ('Playground', 'tenniscourt'),
        ('River', 'river'),
        ('SparseResidential', 'sparseresidential'),
        ('StorageTanks', 'storagetanks')
    ]

    aid_dataset = RSDataset(aid_dataset_path, common_classes, 'AID', data_transform)
    ucm_dataset = RSDataset(ucm_dataset_path, common_classes, 'UCM', data_transform)

    return aid_dataset, ucm_dataset
