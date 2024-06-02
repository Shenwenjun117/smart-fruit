import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from PIL import Image, ImageTk
from torchvision import transforms
import numpy as np
import pickle
import os
from model import VGG16

class FruitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Fruit")

        # 设置窗口的初始大小和位置
        self.root.geometry('500x700+100+100')

        self.model = self.load_model()
        self.classes = ["Apple", "Banana", "Strawberry"]  # 定义类别标签
        self.current_image_path = None

        # 加载颜色直方图模型参数
        self.model_params = self.load_histogram_model_params()
        self.target_mean = self.model_params['target_mean']
        self.target_cov = self.model_params['target_cov']
        self.non_target_mean = self.model_params['non_target_mean']
        self.non_target_cov = self.model_params['non_target_cov']

        # 创建一个主框架来容纳所有内容
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(expand=True)

        # 设置label, button, image label和result label
        self.label = tk.Label(self.main_frame, text="Please select a picture")
        self.label.pack(pady=10)  # 添加一些垂直间距

        self.upload_btn = tk.Button(self.main_frame, text="Upload pictures", command=self.upload_image)
        self.upload_btn.pack(pady=10)  # 添加一些垂直间距

        self.image_label = tk.Label(self.main_frame)
        self.image_label.pack(pady=10)  # 添加一些垂直间距

        self.result_label = tk.Label(self.main_frame, text="")
        self.result_label.pack(pady=10)  # 添加一些垂直间距

        # 创建一个水平框架来容纳“预测正确”和“预测错误”按钮
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=5)

        self.correct_btn = tk.Button(self.button_frame, text="Correct", command=self.correct_prediction)
        self.correct_btn.pack(side=tk.LEFT, padx=5)  # 在水平框架中靠左放置按钮，并添加一些水平间距

        self.wrong_btn = tk.Button(self.button_frame, text="Wrong", command=self.wrong_prediction)
        self.wrong_btn.pack(side=tk.LEFT, padx=5)  # 在水平框架中靠左放置按钮，并添加一些水平间距

    def load_model(self):
        model = VGG16(num_classes=3)
        model_path = os.path.join(os.path.dirname(__file__), 'model_parameters.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def load_histogram_model_params(self):
        params_path = os.path.join(os.path.dirname(__file__), 'model_params.pkl')
        with open(params_path, 'rb') as f:
            return pickle.load(f)

    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        image = image.convert('HSV')
        h, s, v = image.split()
        hist_h = np.histogram(np.array(h), bins=bins[0], range=(0, 256))[0]
        hist_s = np.histogram(np.array(s), bins=bins[1], range=(0, 256))[0]
        hist_v = np.histogram(np.array(v), bins=bins[2], range=(0, 256))[0]
        hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float64)
        hist /= np.sum(hist)
        return hist

    def multivariate_gaussian(self, x, mean, cov):
        size = len(x)
        det = np.linalg.det(cov)
        norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.power(det, 1.0 / 2))
        x_mu = x - mean
        inv = np.linalg.inv(cov)
        result = np.exp(-0.5 * (np.dot(np.dot(x_mu, inv), x_mu.T)))
        return norm_const * result

    def predict_image_class(self, image):
        hist = self.extract_color_histogram(image)
        target_prob = self.multivariate_gaussian(hist, self.target_mean, self.target_cov)
        non_target_prob = self.multivariate_gaussian(hist, self.non_target_mean, self.non_target_cov)
        return target_prob, non_target_prob

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.current_image_path = file_path
            image = Image.open(file_path)
            self.display_image(image)

            # 使用颜色直方图进行预测
            target_prob, non_target_prob = self.predict_image_class(image)
            if target_prob > non_target_prob:
                # 使用VGG16模型进行预测
                prediction, probability = self.predict(image)
                result_text = f"The result is : {self.classes[prediction]} (Probability : {probability:.2f})"
                self.result_label.config(text=result_text)
            else:
                self.result_label.config(text="There is something wrong with the picture, please upload it again.")

    def display_image(self, image):
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def predict(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 转换为Tensor，如果用于PyTorch模型
            transforms.Normalize(mean=[0.7660, 0.6685, 0.6086], std=[0.2478, 0.3280, 0.3922])  # 标准化
        ])
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            return predicted.item(), max_prob.item()

    def correct_prediction(self):
        messagebox.showinfo("Feedback", "Thank you for your feedback!")

    def wrong_prediction(self):
        if self.current_image_path:
            self.show_correct_class_dialog()

    def show_correct_class_dialog(self):
        correct_class_window = tk.Toplevel(self.root)
        correct_class_window.title("Select the right category")
        correct_class_window.geometry('300x200')

        label = tk.Label(correct_class_window, text="Please select the correct category:")
        label.pack(pady=10)

        for idx, class_name in enumerate(self.classes):
            btn = tk.Button(correct_class_window, text=class_name,
                            command=lambda idx=idx: self.save_incorrect_image(idx, correct_class_window))
            btn.pack(pady=5)

    def save_incorrect_image(self, class_idx, window):
        # 保存错误的图像到相应的文件夹
        incorrect_images_dir = "incorrect_images"
        if not os.path.exists(incorrect_images_dir):
            os.makedirs(incorrect_images_dir)

        class_dir = os.path.join(incorrect_images_dir, self.classes[class_idx])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        image_path = os.path.join(class_dir, os.path.basename(self.current_image_path))
        image = Image.open(self.current_image_path)
        image.save(image_path)

        messagebox.showinfo("Feedback", f"The picture was saved to {class_dir}. Thank you for your feedback!")
        window.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = FruitClassifierApp(root)
    root.mainloop()


