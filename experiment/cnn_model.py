import torch.nn as nn

# Define the CNN Model with an additional Conv1D layer

class CNNModel(nn.Module):
    def __init__(self, input_dim,window_size=20, hidden_dim=64, kernel_size=3, device='cpu'):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=1, device=device) # o/p shape: {(I-K+2*P)/strides +1} = window_size
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=kernel_size, padding=1, device=device)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim*4, kernel_size=kernel_size, padding=1, device=device)  # Additional Conv1D layer
        # self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_size, padding=1, device=device)
        self.pool = nn.MaxPool1d(kernel_size=2).to(device)
        # self.fc1 = nn.Linear(hidden_dim*4 * (window_size // 8), 50, device=device)  # Adjust the size according to the output of the conv layers
        self.fc1 = nn.Linear(512 *2, 50, device=device) 
        self.fc2 = nn.Linear(50, 1, device=device)
    
    def forward(self, x):
        # print("Input shape:", x.shape)
        x = nn.ReLU()(self.conv1(x))
        # print("After conv1:", x.shape)
        x = self.pool(x)
        # print("After pool1:", x.shape)
        x = nn.ReLU()(self.conv2(x))
        # print("After conv2:", x.shape)
        x = self.pool(x)
        # print("After pool2:", x.shape)
        x = nn.ReLU()(self.conv3(x))
        # print("After conv3:", x.shape)
        x = self.pool(x)
        # print("After pool3:", x.shape)
        # x = nn.ReLU()(self.conv4(x))
        # print("After conv4:", x.shape)
        # x = self.pool(x)
        # print("After pool4:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        # print("After flatten:", x.shape)
        x = nn.ReLU()(self.fc1(x))
        # print("After fc1:", x.shape)
        x = self.fc2(x)
        # print("After fc2:", x.shape)
        return x
# model = CNNModel(26, 20, hidden_dim=64,kernel_size=3, device='cuda')
if __name__=='__main__':
    import torch
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Assuming your data is in a NumPy array or a PyTorch tensor
    epochs = 1
    num_data_points = 20000
    num_features = 26
    tcn_channels = [64, 128, 256] 
    sequence_length = 20
    batch_size = 32
    num_samples = 1000
    lr = 0.001

    # Create a dummy dataset with 20000 data points and 26 features
    data = np.random.randn(num_data_points, num_features)  # Replace with your actual data

    # Create overlapping sequences
    sequences = []
    for i in range(num_data_points - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])

    # Convert to a PyTorch tensor

    model = CNNModel(26, 20, hidden_dim=64,kernel_size=3, device=device)

    X_dummy = torch.tensor(np.array(sequences), dtype=torch.float32)  # Shape: (19981, 20, 26)

    # Create dummy target values
    y_dummy = torch.randn(X_dummy.shape[0], 1)  # Shape: (19981, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),  lr=lr)


    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0

        # Iterate over batches
        for i in range(0, X_dummy.size(0), batch_size):
            # Get the mini-batch
            X_batch = X_dummy[i:i + batch_size]
            y_batch = y_dummy[i:i + batch_size]

            # Forward pass
            optimizer.zero_grad()
            X_batch = torch.permute(X_batch, (0, 2, 1)) # (batch_size, in_channels, sequence_length)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            break

        #Print average loss for the epoch
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / (num_samples // batch_size):.4f}')
