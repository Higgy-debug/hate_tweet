import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

torch.manual_seed(42)
np.random.seed(42)

smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

X_train_tensor = torch.FloatTensor(X_train_resampled.toarray())  # Convert sparse matrix to array
X_test_tensor = torch.FloatTensor(vectorizer.transform(X_test).toarray())

y_train_tensor = torch.LongTensor(y_train_resampled.values)  # Convert labels to LongTensor for classification
y_test_tensor = torch.LongTensor(y_test.values)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.5):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout_rate, bidirectional=True)

        self.bn = nn.BatchNorm1d(hidden_dim * 2)  

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        
        x = x.unsqueeze(1)  

        lstm_out, (h_n, c_n) = self.lstm(x)

        out = lstm_out[:, -1, :]

        out = self.bn(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out) 
        out = self.fc2(out)

        return out

# Hyperparameter Grid Search
hidden_dims = [64, 128, 256]  
num_layers = [2, 3, 4]  
learning_rates = [0.001, 0.0005, 0.0001]  
dropout_rates = [0.2, 0.3, 0.5]  
batch_sizes = [32, 64, 128]  
epochs = 15

# Track best performance
best_model = None
best_accuracy = 0.0
best_f1 = 0.0
best_params = {}

# Hyperparameter tuning loop
for hidden_dim in hidden_dims:
    for num_layer in num_layers:
        for lr in learning_rates:
            for dropout in dropout_rates:
                for batch_size in batch_sizes:
                    print(f"Training with hidden_dim={hidden_dim}, num_layer={num_layer}, lr={lr}, dropout={dropout}, batch_size={batch_size}")
                    
                    # Initialize the model
                    input_dim = X_train_tensor.shape[1] 
                    output_dim = 2  
                    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers=num_layer, dropout_rate=dropout)

                    class_counts = y_train.value_counts()
                    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
                    class_weights = class_weights / class_weights.sum()  

                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                    for epoch in range(epochs):
                        model.train()  
                        running_loss = 0.0
                        for inputs, labels in train_loader:
                            optimizer.zero_grad()  
                            outputs = model(inputs)  
                            loss = criterion(outputs, labels) 
                            loss.backward()  
                            optimizer.step() 

                            running_loss += loss.item()

                        avg_train_loss = running_loss / len(train_loader)

                    model.eval()  
                    y_pred = []
                    y_true = []
                    y_prob = [] 
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            outputs = model(inputs)
                            probs = torch.nn.functional.softmax(outputs, dim=1)  
                            _, predicted = torch.max(outputs, 1)  
                            y_pred.extend(predicted.cpu().numpy())  
                            y_true.extend(labels.cpu().numpy()) 
                            y_prob.extend(probs.cpu().numpy())  

                    # Calculate accuracy and F1-score
                    accuracy = accuracy_score(y_true, y_pred)
                    f1 = classification_report(y_true, y_pred, output_dict=True)["weighted avg"]["f1-score"]
                    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

                    if f1 > best_f1:
                        best_accuracy = accuracy
                        best_f1 = f1
                        best_model = model
                        best_params = {
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layer,
                            "learning_rate": lr,
                            "dropout_rate": dropout,
                            "batch_size": batch_size
                        }

print("\nBest Model:")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"F1 Score: {best_f1:.4f}")
print("Best Hyperparameters:", best_params)

best_model.eval()  
y_pred_best = []
y_true_best = []
y_prob_best = []  
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = best_model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)  
        _, predicted = torch.max(outputs, 1)  
        y_pred_best.extend(predicted.cpu().numpy()) 
        y_true_best.extend(labels.cpu().numpy())  
        y_prob_best.extend(probs.cpu().numpy())  

# Recalculate accuracy and classification report with best model
accuracy_best = accuracy_score(y_true_best, y_pred_best)
print(f'Accuracy of Best Model: {accuracy_best:.4f}')
print(classification_report(y_true_best, y_pred_best, target_names=["Not Hate", "Hate"]))

#Confusion Matrix
cm_best = confusion_matrix(y_true_best, y_pred_best)
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Hate", "Hate"], yticklabels=["Not Hate", "Hate"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of Best Model')
plt.show()

threshold = 0.4
y_pred_thresholded_best = [1 if prob[1] > threshold else 0 for prob in y_prob_best]  # Apply threshold for class prediction

accuracy_thresholded_best = accuracy_score(y_true_best, y_pred_thresholded_best)
print(f'Accuracy (Threshold {threshold}): {accuracy_thresholded_best:.4f}')
print(classification_report(y_true_best, y_pred_thresholded_best, target_names=["Not Hate", "Hate"]))

y_true_bin = label_binarize(y_true_best, classes=[0, 1])  
roc_auc = roc_auc_score(y_true_bin, y_prob_best, multi_class='ovr')
print(f"AUC-ROC: {roc_auc:.4f}")

# Plot ROC Curve for the Best Model
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), np.array(y_prob_best).ravel())  
roc_auc_val = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Best Model')
plt.legend(loc='lower right')
plt.show()

