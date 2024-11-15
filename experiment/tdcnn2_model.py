import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(channel, channel // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channel // reduction, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        
        # Combine both poolings by summing them (can also concatenate and add an additional conv if desired)
        y = avg_pool + max_pool  
        
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y  # Scale input by the recalibrated weights

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, se_reduction=16):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding='same', dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding='same', dilation=dilation)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.se_block = SEBlock(out_channels, reduction=se_reduction)  # SE Block in each TemporalBlock
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        # print(out.shape)
        out = self.dropout(out)
        out = self.conv2(out)
        # print(out.shape)
        out = self.se_block(out)  # Apply SE Block after conv2
        # print(out.shape)
        if self.residual:
            x = self.residual(x)
        # print(out.shape, x.shape)
        return self.relu(out + x)

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, se_reduction=16):
        super(TCN, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, padding=(kernel_size - 1) * dilation, se_reduction=se_reduction)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FPCASETCN(nn.Module):
    def __init__(self, input_size, tcn_channels, kernel_size=2, se_reduction=16):
        super(FPCASETCN, self).__init__()
        self.tcn = TCN(input_size, tcn_channels, kernel_size, se_reduction)
        self.fc = nn.Linear(tcn_channels[-1], 1)  # Output for RUL prediction

    def forward(self, x):
        tcn_out = self.tcn(x)
        tcn_out = torch.mean(tcn_out, dim=-1)  # Global average pooling
        return self.fc(tcn_out)

# Example usage
# Assuming x_train shape is (batch_size, time_steps, features)
# model = FPCASETCN(x_train.shape[-1], [64, 128, 256])
if __name__=='__main__':
    import torch
    import numpy as np

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

    model = FPCASETCN(input_size=num_features, tcn_channels=tcn_channels)

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
            # print(f"input_shape: {X_batch.shape}")
            X_batch = torch.permute(X_batch, (0, 2, 1)) # (batch_size, in_channels, sequence_length)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            break

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / (num_samples // batch_size):.4f}')
