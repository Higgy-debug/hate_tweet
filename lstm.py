import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Convert to PyTorch tensors

X_train_tensor = torch.FloatTensor(X_train.toarray())  

X_test_tensor = torch.FloatTensor(vectorizer.transform(X_test).toarray())  

y_train_tensor = torch.LongTensor(y_train.values)  
y_test_tensor = torch.LongTensor(y_test.values)

# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)



# Define the LSTM-based Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)  
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        out = self.fc(lstm_out[:, -1, :])
        return out

# Initialize the model
input_dim = X_train_tensor.shape[1]  
hidden_dim = 128  
output_dim = 2  

model = LSTMModel(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  
        outputs = model(inputs) =
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step() 
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    
    scheduler.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}')

model.eval()  
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  
        y_pred.extend(predicted.cpu().numpy())  
        y_true.extend(labels.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_true, y_pred, target_names=["Not Hate", "Hate"]))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Hate", "Hate"], yticklabels=["Not Hate", "Hate"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

