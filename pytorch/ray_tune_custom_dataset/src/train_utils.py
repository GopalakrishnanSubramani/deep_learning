import torch

#training function
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in enumerate(data_loader):
        counter += 1
        image, labels = data
        image, labels= image.to(device), labels.to(device)
        optimizer.zero_grad()

        #forward pass
        outputs = model(image)

        #calculate loss
        loss = criterion(outputs,labels)
        train_running_loss += loss.item()

        #calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds==labels).sum().item()

        #Backprobagation
        loss.backward()
        # Update the optimizer parameters.
        optimizer.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(data_loader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, data_loader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(data_loader.dataset))
    return epoch_loss, epoch_acc
