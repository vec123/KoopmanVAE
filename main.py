import os
import torch
import torch.nn as nn
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from conf.config import KoopmanConfig, DataConfig, ModelDims, LossWeights, TrainConfig
from models.models import KoopmanEncoder, LinearMatrix,ResidualMLP
from models.model_inits import apply_system_init
from training.engine import KoopmanTrainer
from data.loader_factory import DataPipelineFactory

# Load local environment variables (e.g., DATASET_PATH)
load_dotenv()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)

    koopman_cfg = KoopmanConfig(
        exp_name=model_dict['exp_name'],
        checkpoint_dir=model_dict['checkpoint_dir'],
        concat_true=model_dict.get('concat_true', False),
        
        data=DataConfig(**model_dict['data']),
        dims=ModelDims(**model_dict['dims']),
        weights=LossWeights(**model_dict['weights']),
        train=TrainConfig(**model_dict['train'])  # All training params (lr, etc.) handled here
    )
    # Data Pipeline
    # Hydra provides dot-notation (cfg.model.lr)
    # os.getenv handles the local machine infrastructure
    dataset_path = os.getenv("DATASET_W_METEO_PATH")
    
    if not dataset_path:
        raise ValueError("DATASET_PATH not found. Ensure your .env file is set up.")
    
    train_loader, val_loader, scaler = DataPipelineFactory.create_pipeline(
            directories=[dataset_path],         
            config=koopman_cfg.data,             
            batch_size=koopman_cfg.train.batch_size, 
            backend=koopman_cfg.train.backend ,
            num_workers= 5,
            pin_memory = True,
            verbose = True      
        )


    # Model Definitions (System Bundle)
    # Using cfg.model for hyperparameters
    extended_state_dim = koopman_cfg.dims.state_dim + (len(koopman_cfg.data.cyclic_features) * 2)
    dx = extended_state_dim
    dz = koopman_cfg.dims.latent_dim
    du = koopman_cfg.dims.control_dim
    
    # The concatenated latent dimension used by A and the Decoder
    if koopman_cfg.train.concat_true:
        d_total = dz + dx 
    else:
         d_total = dz

    system_bundle = {
        'encoder': KoopmanEncoder(
            state_dim=dx, 
            latent_dim=dz, # Encoder only produces the abstract part
            hidden_dim_int=cfg.model.get('hidden_dim', 64), 
            hidden_depth=cfg.model.get('hidden_depth', 5)
        ),
        
        # Decoder must now accept the concatenated [z, x] vector
        # 'decoder': LinearMatrix(
        #     in_dim=d_total, 
        #     out_dim=dx
        # ),
        'decoder':  ResidualMLP(d_total, dx, [cfg.model.get('hidden_dim', 64)]*cfg.model.get('hidden_depth', 5)),
        
        # The Koopman Operator (A) maps [z, x]_t -> [z, x]_t+1
        'A': LinearMatrix(
            in_dim=d_total,
            out_dim=d_total
        )
    }
    
    apply_system_init(system_bundle, mode='gaussian')

    # Control Logic
    if koopman_cfg.train.encode_control:
        # Effective control dimension is dz if using encoder
        u_latent_dim = dz 
        
        system_bundle['control_encoder'] = nn.Sequential(
            nn.Linear(dx + du, 64),
            nn.ReLU(),
            nn.Linear(64, u_latent_dim)
        )
        system_bundle['control_decoder'] = nn.Sequential(
            nn.Linear(u_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, du)
        )
        # B maps from latent control to the concatenated state space
        system_bundle['B'] = LinearMatrix(u_latent_dim, d_total)
    else:
        # B maps from raw control to the concatenated state space
        system_bundle['B'] = LinearMatrix(du, d_total)



    # 3. External Forcing (Weather)
    if koopman_cfg.dims.forcing_dim > 0:
        system_bundle['E'] = LinearMatrix(koopman_cfg.dims.forcing_dim, d_total)
    # Initialize Trainer
    # engine expects an object with .lr, .horizon, etc. 
    trainer = KoopmanTrainer(
            system_bundle=system_bundle,
            config=koopman_cfg, 
            dataloaders=(train_loader, val_loader)
        )

    # Training Loop
    num_epochs = 10000
    print(f"Starting training on {cfg.device}...")
    print(f"Dataset: {dataset_path}")

    trainer.fit(num_epochs)
    
    # 6. Save Artifacts
    # Hydra automatically creates a unique output folder for every run!
    # Check the 'outputs/' directory after running.
    save_path = "koopman_vae_model.pt"
    torch.save(trainer.models.state_dict(), save_path)
    print(f"Training complete. Model saved to {os.getcwd()}/{save_path}")

if __name__ == "__main__":
    main()