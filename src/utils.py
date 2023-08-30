import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class Standout(nn.Module):
    def __init__(self, previousWeights, alpha, beta, nonlinearity):
        super(Standout, self).__init__()
        self.pi = previousWeights
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nonlinearity # Sigmoid used in the original paper

    def forward(self, inputs, outputs):
        self.p = self.nonlinearity(self.alpha*inputs.matmul(self.pi.t()) + self.beta)
        if(self.training):
            self.mask = sample_mask(self.p)
            return self.mask*outputs
        else:
            return self.p*outputs

class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class ResNet50_DA(nn.Module):
    def __init__(self, num_classes, adaptive_dropout=False, adaptive_bn=False):
        super(ResNet50_DA, self).__init__()
        self.base_model = resnet50(pretrained=True)
        self.adaptive_dropout = adaptive_dropout
        self.adaptive_bn = adaptive_bn
        self.fc1 = nn.Linear(2048, 1024)
        self.out = nn.Linear(1024, num_classes)

        if self.adaptive_bn:
            self.base_model.bn1 = AdaptiveBatchNorm2d(64)

        if self.adaptive_dropout:
            self.std1 = Standout(self.fc1.weight, 0.5, 0.5, nn.Sigmoid())

    def forward(self, x, domain_mask=None):
        features = self.base_model.conv1(x)
        features = self.base_model.bn1(features) if self.adaptive_bn else self.base_model.bn1(features)
        features = self.base_model.relu(features)
        features = self.base_model.maxpool(features)
        features = self.base_model.layer1(features)
        features = self.base_model.layer2(features)
        features = self.base_model.layer3(features)
        features = self.base_model.layer4(features)
        features = self.base_model.avgpool(features)
        features = torch.flatten(features, 1)

        inputs1 = features.clone().detach()
        features = F.relu(self.fc1(features))
        if self.adaptive_dropout:
            features = self.std1(inputs1, features)

        class_output = self.out(features)

        return class_output

def train_test_split_by_class(dataset, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    data_by_class = {}
    for img_path, label in dataset.data:
        if label not in data_by_class:
            data_by_class[label] = []
        data_by_class[label].append((img_path, label))

    train_data, test_data = [], []
    for label, data in data_by_class.items():
        n_test = int(len(data) * test_size)
        test_indices = np.random.choice(len(data), n_test, replace=False)
        for i, item in enumerate(data):
            if i in test_indices:
                test_data.append(item)
            else:
                train_data.append(item)
    return train_data, test_data

def coral(source, target):

    d = source.size(1)

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss

def compute_covariance(input_data):
    n = input_data.size(0)  # batch_size

    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c

def sample_mask(p):
    uniform = torch.Tensor(p.size()).uniform_(0, 1).to(device)
    mask = uniform < p
    return mask.float()

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
