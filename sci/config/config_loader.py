"""Configuration loader for SCI."""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class AbstractionLayerConfig:
    """Configuration for AbstractionLayer."""
    hidden_multiplier: int = 2
    residual_init: float = 0.1
    temperature: float = 0.1
    dropout: float = 0.1
    injection_layers: List[int] = field(default_factory=lambda: [3, 6, 9])


@dataclass
class StructuralEncoderConfig:
    """Configuration for Structural Encoder."""
    num_slots: int = 8
    num_layers: int = 12
    d_model: int = 512
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    abstraction_layer: AbstractionLayerConfig = field(default_factory=AbstractionLayerConfig)


@dataclass
class ContentEncoderConfig:
    """Configuration for Content Encoder."""
    num_layers: int = 12
    d_model: int = 512
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    pooling: str = "mean"  # mean or max


@dataclass
class CausalBindingConfig:
    """Configuration for Causal Binding Mechanism."""
    injection_layers: List[int] = field(default_factory=lambda: [6, 12, 18])
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class PositionalEncodingConfig:
    """Configuration for positional encoding."""
    type: str = "rotary"  # rotary or alibi
    max_length: int = 512
    base: int = 10000  # For RoPE


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Base model
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    d_model: int = 2048  # TinyLlama hidden size
    num_decoder_layers: int = 22  # TinyLlama num layers

    # SCI components
    use_abstraction_layer: bool = True
    use_scl: bool = True
    use_orthogonality_loss: bool = True
    use_causal_binding: bool = True

    # Encoder configurations
    structural_encoder: StructuralEncoderConfig = field(default_factory=StructuralEncoderConfig)
    content_encoder: ContentEncoderConfig = field(default_factory=ContentEncoderConfig)
    causal_binding: CausalBindingConfig = field(default_factory=CausalBindingConfig)
    position_encoding: PositionalEncodingConfig = field(default_factory=PositionalEncodingConfig)

    # Projection dimensions
    projection_dim: int = 2048  # Project SCI encoders to TinyLlama dimension


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "AdamW"
    lr: float = 2e-5
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "cosine"
    num_training_steps: int = 50000
    num_warmup_steps: int = 1000


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    max_epochs: int = 50
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    mixed_precision: bool = True  # Use fp16

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class LossConfig:
    """Loss function weights and parameters."""
    task_weight: float = 1.0
    scl_weight: float = 0.3
    scl_warmup_steps: int = 5000
    ortho_weight: float = 0.1
    eos_weight: float = 2.0
    scl_temperature: float = 0.07


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "scan"
    split: str = "length"  # or "template", "addprim_jump"
    max_length: int = 512
    scl_ratio: float = 0.5  # 50% of batch for SCL pairs
    num_workers: int = 4


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    batch_size: int = 64
    beam_size: int = 1  # Greedy decoding for fair comparison
    max_generation_length: int = 512


@dataclass
class LoggingConfig:
    """Logging configuration."""
    wandb_project: str = "SCI-SCAN"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    log_every_n_steps: int = 100
    checkpoint_every_n_epochs: int = 5


@dataclass
class CheckpointingConfig:
    """Checkpointing configuration."""
    save_dir: str = "checkpoints"
    keep_last_n: int = 3


@dataclass
class SCIConfig:
    """Complete SCI configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)

    # Random seeds for reproducibility
    seed: int = 42


def load_config(config_path: str) -> SCIConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        SCIConfig instance
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Handle _base_ inheritance
    if '_base_' in config_dict:
        base_path = config_path.parent / config_dict.pop('_base_')
        base_config_dict = {}

        if base_path.exists():
            with open(base_path, 'r') as f:
                base_config_dict = yaml.safe_load(f)

        # Merge configs (current overrides base)
        config_dict = _merge_configs(base_config_dict, config_dict)

    # Convert to dataclass
    config = _dict_to_config(config_dict)

    return config


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config."""
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def _dict_to_config(config_dict: Dict[str, Any]) -> SCIConfig:
    """Convert dictionary to SCIConfig dataclass."""
    # Helper function to convert nested dicts
    def convert_section(section_dict: Dict[str, Any], section_class):
        if section_dict is None:
            return section_class()

        kwargs = {}
        for key, value in section_dict.items():
            if isinstance(value, dict):
                # Find the corresponding field type
                field_type = section_class.__dataclass_fields__[key].type
                # Handle Optional types
                if hasattr(field_type, '__origin__'):
                    field_type = field_type.__args__[0]
                kwargs[key] = convert_section(value, field_type)
            else:
                kwargs[key] = value

        return section_class(**kwargs)

    # Convert each section
    config_kwargs = {}

    for section_name, section_class in [
        ('model', ModelConfig),
        ('training', TrainingConfig),
        ('loss', LossConfig),
        ('data', DataConfig),
        ('evaluation', EvaluationConfig),
        ('logging', LoggingConfig),
        ('checkpointing', CheckpointingConfig),
    ]:
        if section_name in config_dict:
            config_kwargs[section_name] = convert_section(
                config_dict[section_name], section_class
            )

    # Add seed if present
    if 'seed' in config_dict:
        config_kwargs['seed'] = config_dict['seed']

    return SCIConfig(**config_kwargs)


def save_config(config: SCIConfig, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: SCIConfig instance
        save_path: Path to save YAML file
    """
    from dataclasses import asdict

    config_dict = asdict(config)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
