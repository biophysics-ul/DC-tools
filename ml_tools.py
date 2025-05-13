'''
1. Data management
'''

import numpy as np
import torch
import pickle
import os
import tkinter
from tkinter import filedialog
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
import torch
import zipfile

class DataTransforms:
    def __init__(self, image_folder_path, batch_size=64, num_workers=0):
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = None
        self.std = None
        self.data_transforms = None

    def calculate_mean_std(self):
        # Define the transformation to convert images to tensors without normalization
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor()
        ])

        # Load the dataset using the SegmentedImages class
        dataset = SegmentedImages(
            image_folder_path=self.image_folder_path,
            transform=transform,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Initialize variables to calculate mean and std
        mean = 0.0
        std = 0.0
        nb_samples = 0

        # Iterate over the dataset to calculate mean and std
        for data in tqdm(loader, total=len(loader), desc = "Calculating metrics"):
            images, _ = data
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

        print("Mean:", mean)
        print("Standard Deviation:", std)

        self.mean = mean
        self.std = std

    def get_transforms(self, train_ok):
        if self.mean is None or self.std is None:
            if train_ok:
                self.calculate_mean_std() 
            else:
                self.mean = torch.tensor([0.5405, 0.5405, 0.5405]) 
                self.std = torch.tensor([0.0881, 0.0881, 0.0881])
            
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Only brightness and contrast for grayscale
                transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), shear=10),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels if required by the model
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())
            ]),
            'eval': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),  # Convert to RGB
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())
            ])
        }
        return self.data_transforms

# Dataset for raw_images + labeled_box_list
class SegmentedImages(Dataset):
    def __init__(self, image_folder_path, transform=None, target_transform=None):
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = []
        self.labels = []

        # Read from zip file
        self.zip_file = zipfile.ZipFile(image_folder_path, 'r')
        for file in self.zip_file.namelist():
            if file.endswith('.tif'):
                label = file.split('/')[0]
                self.image_files.append((file, label))
                if label not in self.labels:
                    self.labels.append(label)

    def __del__(self):
        self.zip_file.close()

    def normalize_image(self, image):
        if image.mode != 'L':
            image = image.convert('L')
        np_image = np.array(image)
        min_val = np.min(np_image)
        max_val = np.max(np_image)
        if min_val == max_val:
            return Image.fromarray(np.zeros_like(np_image).astype('uint8'), 'L')
        np_image = ((np_image - min_val) / (max_val - min_val) * 255).astype('uint8')
        return Image.fromarray(np_image, 'L')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name, label_str = self.image_files[idx]
        with self.zip_file.open(img_name) as image_file:
            raw_image = Image.open(image_file).convert("RGB")
        image = self.normalize_image(raw_image)

        # Convert string label to one-hot encoded label
        label = torch.zeros(len(self.labels))
        label[self.labels.index(label_str)] = 1

        if self.target_transform:
            label = self.target_transform(label)
        
        if self.transform:
            image = self.transform(image)
        return image, label


class UnlabeledImages(Dataset):
    def __init__(self, image_zip_path, transform=None):
        self.zip_path = image_zip_path
        self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
        self.image_files = [f for f in self.zip_file.namelist() if f.endswith('.tif')]
        self.transform = transform

    def normalize_image(self, image):
        image = image.convert("L")
        np_image = np.array(image)
        min_val = np.min(np_image)
        max_val = np.max(np_image)
        if min_val == max_val:
            return Image.fromarray(np.zeros_like(np_image).astype('uint8'), 'L')
        np_image = ((np_image - min_val) / (max_val - min_val) * 255).astype('uint8')
        return Image.fromarray(np_image, 'L')

    def __len__(self):
        return len(self.image_files)

    def process_image(self, image):
        image = self.normalize_image(image)
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        with self.zip_file.open(img_name) as image_file:
            raw_image = Image.open(image_file).convert("RGB")
        image = self.process_image(raw_image)
        return image, img_name

    def __del__(self):
        self.zip_file.close()

def load_model(model_path=None):
    if model_path is None:
        # Open file dialog if no model path is provided
        root = tkinter.Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        model_path = filedialog.askopenfilename(title="Please select the model you would like to use.")
        root.destroy()

    model_name = os.path.basename(model_path)

    # CUDA check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not torch.cuda.is_available():
        print('Cuda is not available. Program will run on CPU, \nso performance will be much slower.')
    else:
        print('Running on cuda')

    # Extract labels
    parts = model_name.split('_')
    try:
        start_index = parts.index("RN") + 1
        labels = parts[start_index:]
    except ValueError:
        labels = []

    print("Labels:", labels)

    # Load the model
    model = torch.jit.load(model_path, map_location='cpu')
    model.to(device)
    model.eval()

    print("Chosen model:", model_name)

    return model, labels, device


def write_to_file(file, elements):
    for i, element in enumerate(elements):
        element_str = str(element)
        if i < len(elements) - 1:
            file.write(element_str.rjust(10) + '\t')
        else:
            file.write(element_str.rjust(10))
    file.write("\n")

def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    try:
        labels = torch.stack(labels, 0)
    except:
        labels = [x for x in labels if x is not None]
    images = torch.stack(images, 0)
    return images, labels

def check_box_list():
    root = tkinter.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    box_list_path = filedialog.askopenfilename(title="Please select the box_list you wish to check.")
    root.destroy()

    with open(box_list_path, "rb") as file:
        box_list = pickle.load(file)

    print("Labels: ")
    print(box_list[-1])  

    label_counts = {}

    print("Cells: ")
    printed_cells_count = 0
    for box in box_list[:-1]:  # Process all boxes except the last item (assumed to be the labels list)
    # Increment the count for the current label in the dictionary
            label_counts[box['label']] = label_counts.get(box['label'], 0) + 1

    # Print details only for the first 5 elements
    if printed_cells_count < 5:
            print(box['cell_name'] + '      ' + box['label'])
            printed_cells_count += 1

    print('Number of cells: ' + str(len(box_list[:-1])))

    # Print the counts for each label
    for label, count in label_counts.items():
            print(f"{label}: {count}")

def choose_image_folder():
    root = tkinter.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    image_folder = filedialog.askopenfilename(
        title="Please select the zipped file containing images.",
        filetypes=[("ZIP files", "*.zip")]
    )
    root.destroy()
    print("Chosen image folder:", image_folder)
    return image_folder

def choose_image_folder_nozip():
    root = tkinter.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    image_file_path = filedialog.askopenfilename(title="Please select the first image in desired folder.", filetypes=[("Image files", ".tif")])
    image_folder = os.path.dirname(image_file_path)
    root.destroy()
    print("Chosen image folder:", image_folder)
    return image_folder

def choose_results_folder():
    root = tkinter.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    image_folder = filedialog.askdirectory(title="Please select the folder you wish to save your results to.")
    root.destroy()
    print("Results will be saved in:", image_folder)
    return image_folder

def get_labels(image_folder_path):
    labels = []
    
    if image_folder_path.endswith('.zip'):
        with zipfile.ZipFile(image_folder_path, 'r') as zip_file:
            for file in zip_file.namelist():
                directory = os.path.dirname(file)
                if directory and directory not in labels:
                    labels.append(directory)
    else:
        labels = [d for d in os.listdir(image_folder_path) if os.path.isdir(os.path.join(image_folder_path, d))]
    print("Labels:", labels)
    return labels

def check_zip_compression(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            for zinfo in z.infolist():
                print(f"{zinfo.filename}: compress_type={zinfo.compress_type}")

'''
2. Training
'''
import os
import tkinter
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from tqdm import tqdm
import datetime
import zipfile
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_model(image_folder_path, results_folder):
    # Validation split (0.5-1)
    val_split = 0.8
    batch_size = 64
    epochs = 10
    # Patience for early stopping
    patience_total = 100
    model_choice = "resnet"
    # Do you want to train the whole model, or just the final classifier?
    train_whole_model = True

    
    current_date = datetime.datetime.now()
    date_str = current_date.strftime("%Y_%m_%d")
    sorted_labels = sorted(get_labels(image_folder_path))
    label_str = '_'.join(sorted_labels)
    model_name = f"{date_str}_M1_RN_{label_str}"
    results_folder_path = results_folder + '/Models/' + model_name

    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    # CUDA check
    if torch.cuda.is_available():
        device = 'cuda'
        print('Running on cuda')
    else:
        device = 'cpu'
        print('Cuda is not available. Program will run on CPU, \nso performance will be much slower.')

    num_classes = len(sorted_labels)  # needed by CustomClassifier

    # Construct model + custom classifier
    if model_choice == "resnet":
        base_model = models.resnet18(weights=None)  # weights = "ResNet18_Weights.IMAGENET1K_V1"
        # Remove the last two layers (AdaptiveAvgPool2d and Linear)
        base_model = nn.Sequential(*list(base_model.children())[:-2])
        # Custom model
        custom_model_top = CustomClassifierResNet(num_classes)
    elif model_choice == "vgg":
        base_model = models.vgg16(weights=None)#weights='VGG16_Weights.IMAGENET1K_V1')
        # Custom model
        custom_model_top = CustomClassifierVGG16(num_classes)
    else:
        print("Please select a valid model_choice")

    # Freeze pre-trained layers
    for param in base_model.parameters():
        param.requires_grad = train_whole_model

    # Construct final model
    model = nn.Sequential(base_model, custom_model_top).to(device)

    with zipfile.ZipFile(image_folder_path, 'r') as zip_file:
        # Get the list of all files and folders in the zip file
        all_files = zip_file.namelist()

        labels = set()
        for file in all_files:
            if '/' in file:
                labels.add(file.split('/')[0])

        sorted_labels = sorted(labels)

        # Count the number of files in each subfolder
        label_freq = Counter()
        for label in sorted_labels:
            label_files = [file for file in all_files if file.startswith(f"{label}/") and not file.endswith('/')]
            label_freq[label] = len(label_files)

    # Calculate the class frequencies
    total_count = sum(label_freq.values())
    class_weights = {k: total_count / v for k, v in label_freq.items()}

    # Normalize the weights so that they sum to the number of classes
    num_classes = len(label_freq)
    class_weights = {k: v * num_classes / total_count for k, v in class_weights.items()}

    # If you want to convert these weights into a tensor for use in PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_tensor = torch.FloatTensor([class_weights[k] for k in sorted(class_weights.keys())]).to(device)

    # Print labels, number of images, and weights
    print("\nLabels | Number | Weights: ")
    for i, label in enumerate(sorted_labels):
        weight = float(weight_tensor[i])
        count = label_freq.get(label, 0)  # Get the count for the label, or default to 0 if it's not found
        print(f"{label} | {count} | {weight:.5f}")

    # Define loss function, optimizer and epochs
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2)

    # Compute mean and std using DataTransforms class
    data_transforms_instance = DataTransforms(image_folder_path)
    data_transforms = data_transforms_instance.get_transforms(train_ok=False)

    # Data loading
    full_dataset = SegmentedImages(
        image_folder_path=image_folder_path,
        transform=data_transforms['train'],
    )
    # Validation split
    train_size = int(val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    label_train_accuracies = []
    label_val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0

    collected_data = []
    confusion_matrices = []

    conf_matrix_folder_path = os.path.join(results_folder_path, 'Confusion_Matrices')
    if not os.path.exists(conf_matrix_folder_path):
        os.makedirs(conf_matrix_folder_path)

    models_folder_path = os.path.join(results_folder_path, 'Models')
    if not os.path.exists(models_folder_path):
        os.makedirs(models_folder_path)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_accuracy, train_label_acc = train_loop(train_dataloader, model, loss_fn, num_classes, optimizer, device)
        val_loss, val_accuracy, val_label_acc, conf_matrix = val_loop(validation_dataloader, model, loss_fn, num_classes, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        label_train_accuracies.append(train_label_acc)
        label_val_accuracies.append(val_label_acc)
        confusion_matrices.append(conf_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
        plt.title(f'Confusion Matrix for Epoch {t+1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plot_path = os.path.join(conf_matrix_folder_path, f"{model_name}_conf_matrix_epoch_{t+1}.png")
        plt.savefig(plot_path)
        plt.close()

        epoch_data = [t+1, train_loss, val_loss, train_accuracy, val_accuracy] + train_label_acc + val_label_acc
        collected_data.append(epoch_data)

        epoch_model_path = os.path.join(models_folder_path, f"{model_name}_epoch_{t+1}")
        # Convert the model to TorchScript
        scripted_model = torch.jit.script(model)
        # Save the entire model (architecture + weights); this already forces it to .eval() by default.
        torch.jit.save(scripted_model, epoch_model_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_total:
            print("Early stopping triggered")
            break

    final_conf_matrix = confusion_matrices[-1]

    plt.figure(figsize=(10, 8))
    sns.heatmap(final_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(results_folder_path, f"{model_name}_conf_matrix_last_epoch.png"))
    plt.close()


    headers = ['Epoch', 'Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'] + \
              [f'Train {label} Acc' for label in sorted_labels] + \
              [f'Val {label} Acc' for label in sorted_labels]


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_title("Loss Plot")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(range(0, epochs + 1, 10))
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accuracies, label="Training Accuracy")
    ax2.plot(val_accuracies, label="Validation Accuracy")
    ax2.set_title("Accuracy Plot")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticks(range(0, epochs + 1, 10))
    ax2.set_yticks(range(0, 101, 10))
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plot_name = f"{model_name}_performance_plots.png"
    plt.savefig(results_folder_path + '/' + plot_name)

    plt.show()

    write_results_to_file(results_folder_path, model_name, collected_data, headers)
    write_confusion_matrices_to_file(results_folder_path, model_name, confusion_matrices, sorted_labels)

    # Plot per-label accuracies
    # Calculate the number of rows needed
    num_rows = (num_classes + 1) // 2

    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 4 * num_rows))
    axs = axs.flatten()

    for i in range(num_classes):
        label = sorted_labels[i]
        train_label_acc = [epoch_acc[i] for epoch_acc in label_train_accuracies]
        val_label_acc = [epoch_acc[i] for epoch_acc in label_val_accuracies]
        
        axs[i].plot(train_label_acc, label="Training Accuracy")
        axs[i].plot(val_label_acc, label="Validation Accuracy")
        axs[i].set_title(f"Accuracy for Label: {label}")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Accuracy (%)")
        axs[i].set_ylim(0, 100)
        axs[i].set_xticks(range(0, epochs + 1, 10))
        axs[i].set_yticks(range(0, 101, 10))
        axs[i].legend()
        axs[i].grid(True)

    # If the number of classes is odd, remove the empty subplot
    if num_classes % 2 != 0:
        fig.delaxes(axs[-1])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plot_name_lab = f"{model_name}_label_performance.png"
    plt.savefig(results_folder_path + '/' + plot_name_lab)
    plt.show()


    # Save the model
    # Convert the model to TorchScript
    scripted_model = torch.jit.script(model)
    # Save the entire model (architecture + weights); this already forces it to .eval() by default.
    torch.jit.save(scripted_model, results_folder_path + '/' + model_name)

    # Check if it saved correctly
    loaded_scripted_model = torch.jit.load(results_folder_path + '/' + model_name)

    # Comparing models
    def compare_models(model1, model2):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                return False
        return True

    if compare_models(loaded_scripted_model, scripted_model):
        print("Saving/loading works correctly.")

# Top layers (classifier) for our VGG16 model
class CustomClassifierVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomClassifierVGG16, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Top layers (classifier) for our ResNet model
class CustomClassifierResNet(nn.Module):
    def __init__(self, num_classes, noise_std=0.1, dropout_prob=0.3):
        super(CustomClassifierResNet, self).__init__()
        # Flatten layer
        self.flatten = nn.Flatten()
        # First fully connected layer followed by ReLU activation
        self.fc1 = nn.Linear(512 * 7 * 7, 256)#self.fc1 = nn.Linear(512 * 7 * 7, 256)  # Adjust the input dimension based on your specific use case
        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Noise standard deviation
        self.noise_std = noise_std

        # Second fully connected layer (output layer)
        self.fc2 = nn.Linear(256, num_classes)#self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

# Training loop definition
def train_loop(dataloader, model, loss_fn, num_labels, optimizer, device):
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    total_loss = 0
    correct = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # Initialize per-label correct predictions and total count
    label_correct = [0] * num_labels
    label_total = [0] * num_labels

    # Initialize tqdm progress bar
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)  # Move data to CUDA if available
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Compute accuracy
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        # Update per-label accuracy
        for i in range(len(y)):
            true_label = y[i].argmax().item()
            pred_label = pred[i].argmax().item()
            label_total[true_label] += 1
            if true_label == pred_label:
                label_correct[true_label] += 1

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = total_loss / num_batches
    train_accuracy = 100 * correct / size
    print(f"Accuracy: {(train_accuracy):>0.1f}%, Avg loss: {avg_train_loss:>8f} \n")

    label_accuracies = [100 * label_correct[i] / label_total[i] if label_total[i] > 0 else 0 for i in range(num_labels)]

    return avg_train_loss, train_accuracy, label_accuracies

# Test loop definition
def val_loop(dataloader, model, loss_fn, num_labels, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    all_preds = []
    all_labels = []

    label_correct = [0] * num_labels
    label_total = [0] * num_labels

    # Initialize tqdm progress bar
    with torch.no_grad():
        for X, y in tqdm(dataloader, total=len(dataloader), desc="Validation"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            for i in range(len(y)):
                true_label = y[i].argmax().item()
                pred_label = pred[i].argmax().item()
                label_total[true_label] += 1
                if true_label == pred_label:
                    label_correct[true_label] += 1

            all_preds.extend(pred.argmax(1).tolist())
            all_labels.extend(y.argmax(1).tolist())

    avg_val_loss = val_loss / num_batches
    val_accuracy = 100 * correct / size
    print(f"Accuracy: {(val_accuracy):>0.1f}%, Avg loss: {avg_val_loss:>8f} \n")

    label_accuracies = [100 * label_correct[i] / label_total[i] if label_total[i] > 0 else 0 for i in range(num_labels)]
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_labels))

    return avg_val_loss, val_accuracy, label_accuracies, conf_matrix

def write_results_to_file(results_folder_path, model_name, collected_data, headers):
    # Determine the maximum width for each column
    col_widths = [max(len(header), 12) for header in headers]

    # Format and write the headers and data
    data_file_path = os.path.join(results_folder_path, f"{model_name}_training_data.dat")
    with open(data_file_path, 'w') as f:
        # Write the headers with tabs
        header_line = '\t'.join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers))
        f.write(header_line + '\n')

        # Write the data with tabs
        for i, epoch_data in enumerate(collected_data):
            data_line = '\t'.join(
                f"{data:<{col_widths[i]}.6f}" if isinstance(data, float) else f"{str(data):<{col_widths[i]}}"
                for i, data in enumerate(epoch_data)
            )
            f.write(data_line + '\n')


def write_confusion_matrices_to_file(results_folder_path, model_name, confusion_matrices, sorted_labels):
    conf_matrix_file_path = os.path.join(results_folder_path, f"{model_name}_confusion_matrices.dat")
    with open(conf_matrix_file_path, 'w') as f:
        for epoch, conf_matrix in enumerate(confusion_matrices):
            f.write(f"Epoch {epoch+1} Confusion Matrix:\n")
            f.write('\t' + '\t'.join(sorted_labels) + '\n')
            for i, row in enumerate(conf_matrix):
                f.write(sorted_labels[i] + '\t' + '\t'.join(map(str, row)) + '\n')
            f.write("\n")


'''
3. Inference
'''

import torch
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import zipfile
import re



def inference(image_folder, results_folder, model, labels, device):
    # Create dataset and dataloader
    experiment_folder_path = os.path.dirname(image_folder) 
    experiment_name = os.path.basename(experiment_folder_path)
    results_folder_path = results_folder + '/' + experiment_name
    label_str = '_'.join(labels)

    with zipfile.ZipFile(image_folder, 'r') as zip_file:
        # Compute mean and std using DataTransforms class
        data_transforms_instance = DataTransforms(image_folder)
        data_transforms = data_transforms_instance.get_transforms(train_ok=False)

        unlabeled_dataset = UnlabeledImages(image_folder, transform = data_transforms['eval'])
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size = 64, collate_fn=custom_collate)

        ###Inference###
        inference_results = []

        # Calculate the total number of batches
        total_batches = len(unlabeled_dataloader)

        # Infer the labels for the unlabelled images and update the box list. Track updates with tqdm.
        for images, img_names in tqdm(unlabeled_dataloader, total=total_batches, desc='Inference'):
            # Move the batch of images to the device
            images = images.to(device)
            with torch.no_grad():
                # Forward pass for the whole batch
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                confidences, preds = torch.max(probs, 1)

            # Loop through the batch to update the boxes
            for i in range(len(preds)):
                inference_results.append({
                "image_name": img_names[i],
                "n": 0,
                "label": labels[preds[i].item()],
                "confidence": confidences[i].item()
                })


    save_classified_images(image_folder, results_folder, inference_results, labels)
    save_classification_metrics(image_folder, results_folder,inference_results, labels)



def save_classified_images(image_folder, results_folder, inference_results, labels):
    experiment_folder_path = os.path.dirname(image_folder) 
    experiment_name = os.path.basename(experiment_folder_path)
    results_folder_path = results_folder + '/' + experiment_name
    label_str = '_'.join(labels)
    classified_images_path = results_folder_path + '/class_imgs_' + experiment_name + '_' + label_str

    if not os.path.exists(classified_images_path):
        os.makedirs(classified_images_path)

    for label in labels:
        if not os.path.exists(f"{classified_images_path}/{label}"):
            os.makedirs(f"{classified_images_path}/{label}")

    label_counts = {label: 0 for label in labels}

    with zipfile.ZipFile(image_folder, 'r') as zip_file:
        for cell in tqdm(inference_results, desc='Saving images'):
            with zip_file.open(cell["image_name"]) as image_file:
                file_data = image_file.read()
                file_bytes = np.frombuffer(file_data, np.uint8)
                raw_image = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)
                
                if raw_image is None:
                    print(f"Warning: Could not read image {cell['image_name']}")
                    continue

                output_file_name = cell["image_name"]
                output_path = os.path.join(classified_images_path, cell["label"], output_file_name)

                if not cv.imwrite(output_path, raw_image):
                    raise Exception(f"Could not write image to {output_path}")  # imwrite fails silently, this makes it loud.
    
                label = cell['label']
                label_counts[label] += 1

    print("Images saved.")

def save_classification_metrics(image_folder, results_folder, inference_results, labels=False):
    experiment_folder_path = os.path.dirname(image_folder) 
    experiment_name = os.path.basename(experiment_folder_path)
    results_folder_path = results_folder + '/' + experiment_name
    label_str = '_' + '_'.join(labels) if labels else ""
    data_file_name = "class_data_" + experiment_name + label_str + ".txt"
    data_file_path = os.path.join(results_folder_path + '/', data_file_name)

    with open(data_file_path, 'w') as dat_file:
        #dat_file.write("$BEGINDATA\n")  # Add BEGINDATA line for further analysis (OpenShapeOut)

        headers=['img_name', 'label', 'confidence']#headers = ['img_name', 'timestamp', 'n', 'area_pix', 'defor', 'cx_pix', 'cw_pix', 'ch_pix', 'perim_pix','mean_pix', 'stdev_pix', 'label']
        write_to_file(dat_file, headers)
        for cell in tqdm(inference_results, desc='Saving image data'):
            '''
            match = re.search(r'(\d{6})', cell['image_name'])
            n = cell['image_name'].split('-')[-1] if '-' in cell['image_name'] else "n.tif"
            img_name = match.group(1)+"-" + n if match else cell['image_name']
            '''
            img_name=img_name = cell['image_name'][-12:]
            label = cell['label']
            confidence = cell['confidence']
            data_elements = [img_name, label, f"{confidence:.4f}"]
            write_to_file(dat_file, data_elements)
