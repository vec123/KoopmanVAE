from models.model_backbones import *
from models.models import (
    BaseKoopmanEncoder, 
    LinearMatrix, 
    BlockDiagonal,
      LearnableComplexBlockDiagonal, 
      LinearQuadraticOperator)

def get_encoder(cfg, dx, dz):
    """
    Modular Factory for switching encoding strategies.
    """
    etype = cfg.train.encoder_type
    hidden = cfg.dims.hidden_dim
    
    if etype == "resmlp":
        backbone = ResidualMLP(dx, hidden, [hidden]*3)
        feat_dim = hidden
    elif etype == "gru":
        backbone = GRUBackbone(dx, hidden)
        feat_dim = hidden
    elif etype == "transformer":
        backbone = TransformerBackbone(dx, d_model=hidden)
        feat_dim = hidden
    else:
        raise ValueError(f"Unknown encoder type: {etype}")

    return BaseKoopmanEncoder(
        backbone=backbone,
        feature_dim=feat_dim,
        latent_dim=dz,
        stochastic=cfg.train.stochastic
    )


def get_decoder(cfg, d_total, dx):
    """
    Modular Factory for switching decoding strategies.
    """
    # getattr(object, "attr_name", default_value)
    m_type = getattr(cfg.train, 'decoder_type', 'linear') 
    
    hidden = cfg.dims.hidden_dim
    depth = cfg.dims.hidden_depth

    if m_type == 'linear':
        return LinearMatrix(in_dim=d_total, out_dim=dx)
    
    elif m_type == 'resmlp':
        return ResidualMLP(
            in_dim=d_total, 
            out_dim=dx, 
            hidden_channels=[hidden] * depth
        )
    else:
        raise ValueError(f"Unknown decoder type: {m_type}")

def get_koopman_operator(cfg, d_total):
    """
    Factory for the Dynamics Function/Matrix (A)
    """
    m_type = getattr(cfg.train, 'operator_type', 'linear')
    hidden = cfg.dims.hidden_dim
    depth = cfg.dims.hidden_depth

    if m_type == 'linear':
        return LinearMatrix(in_dim=d_total, out_dim=d_total)
    
    elif m_type == 'complex_diag':
        # Ensure latent dim + state dim is even
        if d_total % 2 != 0:
            print(f"Warning: d_total ({d_total}) is odd. ")
        return BlockDiagonal(dim=d_total)
    

    elif m_type == 'learnable_complex_diag':
        # This is the new learnable complex block diagonal dependent on input
        #For now fix the backbone 
        #bb_type = getattr(cfg.train, 'dynamic_backbone_type', 'mlp')
        bb_type = "mlp"
        if bb_type == 'mlp':
            backbone = ResidualMLP(d_total, hidden, [hidden] * 2)
            feat_dim = hidden
        elif bb_type == 'transformer':
            backbone = TransformerBackbone(d_total, d_model=hidden)
            feat_dim = hidden
        else:
            raise ValueError(f"Unknown dynamic backbone: {bb_type}")

        return LearnableComplexBlockDiagonal(
            backbone=backbone, 
            feature_dim=feat_dim, 
            latent_dim=d_total
        )
    
    elif m_type == 'resmlp':
        return ResidualMLP(
            in_dim=d_total, 
            out_dim=d_total, 
            hidden_channels=[d_total] * 2
        )
    elif m_type == "quadratic":
         return LinearQuadraticOperator(dim=d_total)

    else:
        raise ValueError(f"Unknown operator type: {m_type}")