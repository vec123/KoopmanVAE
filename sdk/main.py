from core.models import KoopmanEncoder, ResidualMLP, KoopmanOperator
from training.engine import KoopmanTrainer

class Config:
    lr = 1e-3
    horizon = 24
    latent_dim = 32
    beta_kl = 1e-6
    use_ema = True

def main():
    # 1. Setup Models
    system = {
        'encoder': KoopmanEncoder(input_dim=7, latent_dim=32),
        'decoder': ResidualMLP(input_dim=32, output_dim=7, hidden_list=[128, 128]),
        'A': KoopmanOperator(32, 32),
        'B': KoopmanOperator(1, 32), # Control
        'E': KoopmanOperator(1, 32)  # External forcing
    }

    # 2. Initialize Trainer (Dataloaders assumed created via data module)
    trainer = KoopmanTrainer(
        system_bundle=system,
        config=Config(),
        dataloaders=(train_loader, val_loader)
    )

    # 3. Loop
    for epoch in range(500):
        train_loss = trainer.train_epoch()
        print(f"Epoch {epoch}: Loss {train_loss:.4f}")

if __name__ == "__main__":
    main()