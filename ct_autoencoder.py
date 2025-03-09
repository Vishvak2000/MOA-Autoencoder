import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Randomize non-randomly.
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Use best processing unit available.
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(42)
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Initializes the VAE model with variable hidden layers.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dims (list[int]): List of dimensions for each hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.encoder_layers = nn.ModuleList()
        encoder_dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            self.encoder_layers.append(nn.ReLU())
        self.encoder_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        decoder_dims = [latent_dim] + hidden_dims[::-1] #reverse hidden dims for decoder
        for i in range(len(hidden_dims)):
            self.decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            self.decoder_layers.append(nn.ReLU())
        self.decoder_output = nn.Linear(hidden_dims[0], input_dim)


        # Intermediate values for retrieval.
        self.mu = None
        self.logvar = None

    def encode(self, x):
        """
        Encodes the input into the mean and log variance of the latent
        distribution.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: Mean (mu) and log variance (logvar) of the latent
                   distribution.
        """
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h))
        self.mu = self.encoder_mu(h)
        self.logvar = self.encoder_logvar(h)
        return self.mu, self.logvar


    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the latent distribution to enable gradient-based
        training.

        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Log variance of the latent distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decodes the latent vector into the reconstructed input.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        h = z
        for layer in self.decoder_layers:
            h = F.relu(layer(h))
        #sigmoid for binary data. For real valued data, no sigmoid or tanh.
        return torch.sigmoid(self.decoder_output(h))

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: Reconstructed input, mean, and log variance of the latent
                   distribution.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

def loss_function(reconstruction_x, x, mu, logvar):
    """
    VAE loss function.

    Args:
        reconstruction_x: Reconstructed input.
        x: Original input.
        mu: Mean of the latent distribution.
        logvar: Log variance of the latent distribution.

    Returns:
        Total loss.
    """
    # Using MSE loss.
    reconstruction_loss = F.mse_loss(reconstruction_x, x, reduction='sum')

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + kl_divergence

def train_model(model, train_loader, test_loader, optimizer, criterion, n_epochs, iteration, save_path='data/model_checkpoints', **kwargs):
    """
    Trains an autoencoder model.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        model (nn.Module): The autoencoder model.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        n_epochs (int): Number of training epochs.
        save_path (str): Path to save model checkpoints.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_losses = []

    # Save checkpoints at these epochs
    checkpoint_epochs = [n_epochs // 10, n_epochs // 2, n_epochs]

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1} of {n_epochs}")
        model.train()
        running_loss = 0.0

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            outputs, mu, logvar = model(data)
            model.mu = mu
            model.logvar = logvar

            # Reshape the output to match the input shape if necessary.
            data = data.view(data.size(0), -1) # Flatten the input
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss
        train_error = test_model(model, train_loader)
        test_error = test_model(model, test_loader)

        train_losses.append([epoch_loss, train_error, test_error])

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_error:.4f}")
        print(f"Epoch {epoch+1}/{n_epochs}, Test Loss: {test_error:.4f}")

        # Save checkpoints
        if (epoch + 1) in checkpoint_epochs:
            checkpoint_path = os.path.join(save_path, f"autoencoder_{iteration}_epoch_{epoch+1}.pth")
            torch.save({
                'model': model,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

    return train_losses

def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data in test_loader:
            data_true = data.to(device)
            data_pred, mu, log_var = model(data_true)
            loss = loss_function(data_pred, data_true, mu, log_var)
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)

##### MAIN FUNCTION

# def main(n_iter=100, batch_size = 1024):
# Get annotated dataset and stratify by treatment group (control or not).

batch_size = 1024
n_iter = 10

moa_data = pd.read_csv('data/lish_moa_annotated.csv')
treated_data = moa_data[moa_data['cp_type'] == 'trt_cp']
control_data = moa_data[moa_data['cp_type'] == 'ctl_vehicle']

# Capture meta indices from start of table.
meta_indices = [
    "sig_id",
    "drug_id",
    "training",
    "cp_type",
    "cp_time",
    "cp_dose"
]

# Capture features from prefix values of columns.
expression_indices = list(filter(lambda col: col.startswith('g-'), moa_data.columns))
viability_indices = list(filter(lambda col: col.startswith('c-'), moa_data.columns))
feature_indices = expression_indices + viability_indices

X = moa_data[feature_indices]
y = moa_data[meta_indices]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Scale data according to training dataset. Apply training scale to test
# dataset. This does not transfer underlying knowledge, it's more for
# normalization, so they're still split!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_train_scaled.astype(np.float32)

# Tailor speed to number of workers.
active_cpus = os.cpu_count() or 1

# Create data loaders.
train_loader = torch.utils.data.DataLoader(
    X_train_scaled,
    batch_size=batch_size,
    shuffle=True,
    num_workers=active_cpus)

test_loader = torch.utils.data.DataLoader(
    X_test_scaled,
    batch_size=batch_size,
    shuffle=False,
    num_workers=active_cpus)

input_dim = len(feature_indices)

train_losses = []
test_losses = []
for i in range(n_iter):
    print(f"Starting iteration {i+1} of {n_iter}...")
    linear = np.random.choice([True, False])
    n_hidden_layers = np.random.randint(1, 4) # 1 to 3 hidden layers
    latent_dim = np.random.randint(10, 100)
    if linear:
        # Linearly decay the hidden layers to the latent space.
        points = np.linspace(input_dim, latent_dim, n_hidden_layers + 2)
        hidden_layers = list(map(lambda point: int(point), points[1:-1]))
    else:
        # Logarithmically decay the hidden layers to the latent space.
        log_points = np.linspace(np.log(input_dim), np.log(latent_dim), n_hidden_layers + 2)
        hidden_layers = list(map(lambda l: int(np.exp(l)), log_points))[1:-1]

    variational_autoencoder = VariationalAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_layers,
        latent_dim=latent_dim
    )
    variational_autoencoder.to(device)

    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(variational_autoencoder.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

    train_losses.append(
        train_model(variational_autoencoder, train_loader, test_loader, optimizer, criterion, 500, n_iter, 'data/ct_vae')
    )
    test_losses.append(
        test_model(variational_autoencoder, test_loader)
    )
    torch.save(
        variational_autoencoder, f'ct_vae_models/{n_iter}-model.pt'
    )