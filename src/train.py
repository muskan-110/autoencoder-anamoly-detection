import torch
from torch.utils.data import DataLoader
from models.cnn_autoencoder import CNNAutoencoder
from datasets.fashion_mnist import get_fashion_mnist

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds, _ = get_fashion_mnist(normal_class=7)
loader = DataLoader(train_ds, batch_size=128, shuffle=True)

model = CNNAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    loss_sum = 0
    for x, _ in loader:
        x = x.to(device)
        loss = criterion(model(x), x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    print(f"Epoch {epoch+1}: {loss_sum/len(loader):.4f}")

torch.save(model.state_dict(), "results/model.pth")
