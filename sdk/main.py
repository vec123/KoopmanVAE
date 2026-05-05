import os
import torch
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# SDK Imports
from models.models import KoopmanEncoder, LinearMatrix
from training.engine import KoopmanTrainer
from data.loader_factory import DataPipelineFactory

# Load local environment variables (e.g., DATASET_PATH)
load_dotenv()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Resolve local paths and settings
    # Hydra provides dot-notation (cfg.model.lr)
    # os.getenv handles the local machine infrastructure
    dataset_path = os.getenv("DATASET_PATH")
    
    if not dataset_path:
        raise ValueError("DATASET_PATH not found. Ensure your .env file is set up.")

    #  Data Pipeline
    # combine model params and infrastructure params for the factory
    factory_config = OmegaConf.to_container(cfg.model, resolve=True)
    factory_config['backend'] = cfg.backend
    
    train_loader, val_loader, scaler = DataPipelineFactory.create_pipeline(
        directories=[dataset_path], 
        config=factory_config
    )

    # Model Definitions (System Bundle)
    # Using cfg.model for hyperparameters
    system_bundle = {
        'encoder': KoopmanEncoder(
            input_dim=cfg.model.state_dim, 
            latent_dim=cfg.model.latent_dim
        ),
        'decoder': LinearMatrix(
            latent_dim=cfg.model.latent_dim, 
            output_dim=cfg.model.state_dim
        ),
        'A': LinearMatrix(cfg.model.latent_dim, cfg.model.latent_dim),
    }

    if cfg.model.control_dim > 0:
        system_bundle['B'] = LinearMatrix(cfg.model.control_dim, cfg.model.latent_dim)
    if cfg.model.forcing_dim > 0:
        system_bundle['E'] = LinearMatrix(cfg.model.forcing_dim, cfg.model.latent_dim)

    # Initialize Trainer
    # Your engine expects an object with .lr, .horizon, etc. 
    # Hydra's DictConfig already supports this!
    trainer = KoopmanTrainer(
        system_bundle=system_bundle,
        config=cfg.model, 
        dataloaders=(train_loader, val_loader),
        device=cfg.device
    )

    # Training Loop
    num_epochs = 100
    print(f"Starting training on {cfg.device}...")
    print(f"Dataset: {dataset_path}")

    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

    # 6. Save Artifacts
    # Hydra automatically creates a unique output folder for every run!
    # Check the 'outputs/' directory after running.
    save_path = "koopman_vae_model.pt"
    torch.save(trainer.models.state_dict(), save_path)
    print(f"Training complete. Model saved to {os.getcwd()}/{save_path}")

if __name__ == "__main__":
    main()