import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据转换，包括将PIL Image转换为Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据需要调整大小
    transforms.ToTensor()
])

# 加载你的数据集
# path = 'D:\\Desktop\\Learning\\Intelligent Software Implementation\\software\\image\\Wrong\\train_set'
path = 'D:\\Desktop\\Learning\\Intelligent Software Implementation\\software\\image\\Right\\train'

dataset = datasets.ImageFolder(root= path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


def calculate_mean_std(dataloader):
    # 用于累积所有样本的均值和方差
    mean = 0.0
    var = 0.0
    n_samples = 0
    for data, _ in dataloader:
        batch_samples = data.size(0)  # batch size (the last batch can have smaller size!)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std = torch.sqrt(var / n_samples)
    return mean, std

mean, std = calculate_mean_std(dataloader)
print(f"Mean: {mean}")
print(f"Std: {std}")
