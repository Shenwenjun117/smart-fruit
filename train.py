import random

import numpy as np
import os
import torch
import torch.nn as nn
from collections import Counter
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import VGG16
from torchsummary import summary
from torch.utils.data import random_split
from preprocess import SquarePad
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


class RemapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, new_label):
        self.dataset = dataset
        self.new_label = new_label

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, self.new_label

    def __len__(self):
        return len(self.dataset)

def data_loader(data_dir, batch_size=32, img_size=224):


    # 定义数据转换（分别计算均值和标准差）
    mean = [0.7660, 0.6685, 0.6086]
    std = [0.2478, 0.3280, 0.3922]


    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation((-180, 180)),  # 随机旋转±180度
        transforms.ColorJitter(brightness=0.5),  # 随机调整亮度
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.ToTensor(),  # 将 PIL Image 或 numpy.ndarray 转换为 tensor
        transforms.Normalize(mean=mean, std=std),  # 标准化
    ])
    transform_test = transforms.Compose([
        # SquarePad(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # 转换为Tensor，如果用于PyTorch模型
        transforms.Normalize(mean=mean, std=std),  # 标准化
    ])



    # 加载训练集和测试集
    '''当你使用 ImageFolder 类来加载数据时，它会自动将文件夹名称作为类别标签，并将这些类别映射为整数。
    例如，如果你有三个文件夹分别命名为 "Apple"、"Banana" 和 "Strawberry"，ImageFolder 会自动为这些类别分配一个整数标签，比如 0、1、2。'''

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Right','train'), transform=transform)
    validation_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Right','validation'), transform=transform_test)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Right','test'), transform=transform_test)




    '''创建 DataLoader'''
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, validation_loader, test_loader , test_dataset


def get_class_distribution(loader):
    class_counts = Counter()
    for _, labels in loader:
        class_counts.update(labels.numpy())
    return class_counts

if __name__ == '__main__':

    # 检查一些图像数据
    # print("Train class distribution:", get_class_distribution(train_loader))
    # print("Validation class distribution:", get_class_distribution(validation_loader))
    # print("Test class distribution:", get_class_distribution(test_loader))
    # for images, labels in train_loader:
    #     print(images.shape)  # 应显示 [batch_size, 3, 224, 224]
    #     print(labels)  # 显示标签
        # break
    # for images, labels in test_loader:
    #     print(images.shape)  # 应显示 [batch_size, 3, 224, 224]
    #     print(labels)  # 显示标签
    #     break

    '''Device configuration'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device) # cuda

    '''Hyperparameters'''
    num_classes = 3
    num_epochs = 70
    batch_size = 10
    learning_rate = 0.001

    model = VGG16(num_classes).to(device)
    # print(model)
    # summary(model, (3, 224, 224))
    '''Loss and optimizer'''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ''''''
    path = 'D:\\Desktop\\Learning\\Intelligent Software Implementation\\software\\image'
    train_loader, validation_loader, test_loader, test_dataset = data_loader(path, batch_size)
    if os.path.exists('model_parameters.pth'):
        try:
            # 尝试加载模型参数
            model.load_state_dict(torch.load('model_parameters.pth'))
            print("模型参数加载成功，将不执行训练。")
            model.eval()  # 设置模型为评估模式

        except Exception as e:
            print("加载模型参数时出错:", e)
            print("将开始训练模型。")
            '''Train model'''
            total_step = len(train_loader)
            for epoch in range(num_epochs):
                model.train()
                for i, (images, labels) in enumerate(train_loader):
                    # Move tensors to the configured device
                    images = images.to(device)
                    labels = labels.to(device)
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                         loss.item()))
                # Validation
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in validation_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        del images, labels, outputs

                    print('Accuracy of the network on the validation images: {} %'.format(100 * correct / total))

            '''Preserve model'''
            torch.save(model.state_dict(), 'model_parameters.pth')
    else:
        print("未找到模型参数文件，将开始训练模型。")
        '''Train model'''
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                     loss.item()))
            # Validation
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in validation_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    del images, labels, outputs

                print('Accuracy of the network on the validation images: {} %'.format(100 * correct / total))

        '''Preserve model'''
        torch.save(model.state_dict(), 'model_parameters.pth')

    '''Test'''
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            del images, labels, outputs

        # Calculate accuracy
        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Generate the confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Display the confusion matrix using seaborn
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Print classification report
        class_report = classification_report(all_labels, all_predictions, target_names=test_dataset.classes)
        print('Classification Report:')
        print(class_report)



