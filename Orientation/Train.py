import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


# Constants
i2c = {0: '0', 1: '135', 2: '180', 3: '224', 4: '270', 5: '315', 6: '45', 7: '90'}
classes_num = len(i2c)

train_path = '../Data/Orientation/train/'
test_path = '../Data/Orientation/test/'
val_path = '../Data/Orientation/valid/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imgsz = 640
batch = 32 
epochs = 20

# Preparing data
train_trans = T.Compose([
    T.ToTensor(),
    T.Resize(imgsz),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

eval_trans = T.Compose([
    T.ToTensor(),
    T.Resize(imgsz),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = ImageFolder(train_path, train_trans)
testset = ImageFolder(test_path, eval_trans)
valset = ImageFolder(val_path, eval_trans)


train_loader = DataLoader(trainset, batch, True)
test_loader = DataLoader(testset, batch, True)
val_loader = DataLoader(valset, batch, True)



from torchvision.models import resnet50, ResNet50_Weights
def get_resnet50():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    for p in model.parameters():
        p.requires_grad = False
        
    model.fc = nn.Linear(model.fc.in_features,classes_num)
    return model


from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
def get_efficientnet_b4():
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    for p in model.parameters():
        p.requires_grad = False
        
    model.classifier[1] = nn.Linear(1792,classes_num)
    
    return model


from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
def get_mobilenet_larg():
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    
    for p in model.parameters():
        p.requires_grad = False
        
    model.classifier[3] = nn.Linear(1280,classes_num)
    return model



# --------Train loop---------

def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, predicted = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        print('-' * 50)
        
        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), './Weights/best_model.pth')
            print(f'New best model saved with accuracy: {best_acc:.4f}')
    
    print(f'Training completed. Best validation accuracy: {best_acc:.4f}')
    return history




# -------------------------------------------------------------------------------
# model = get_efficientnet_b4()
# model = get_mobilenet_larg()
model = get_resnet50()

history = train_model(model,train_loader,val_loader,epochs)
