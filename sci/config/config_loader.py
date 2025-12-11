"""Configuration loader for SCI."""

import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

# MEDIUM #55: Use logging instead of print
logger = logging.getLogger(__name__)


@dataclass
class AbstractionLayerConfig:
    """Configuration for AbstractionLayer.
    
    Higher dropout (0.2) for this novel module following NeuroGen pattern.
    """
    hidden_multiplier: int = 2
    residual_init: float = 0.1
    temperature: float = 0.1
    dropout: float = 0.2  # NeuroGen: Higher dropout for novel structural module
    injection_layers: List[int] = field(default_factory=lambda: [3, 6, 9])


@dataclass
class SlotAttentionConfig:
    """Configuration for Slot Attention.
    
    Parameters:
        num_slots: Number of slots for attention (passed from StructuralEncoderConfig)
        num_iterations: Number of iterative refinement steps (NeuroGen: 6 optimal)
        epsilon: Small value for numerical stability in softmax (default: 1e-8)
        hidden_dim: Hidden dimension for slot MLP (default: None, uses d_model)
    """
    num_iterations: int = 6  # NeuroGen transfer: 6 iterations for complex compositions
    epsilon: float = 1e-8
    hidden_dim: int = None  # #100 FIX: Add hidden_dim for completeness
    # #103 NOTE: num_slots is passed from StructuralEncoderConfig, not stored here


@dataclass
class EdgePredictorConfig:
    """Configuration for Edge Predictor in Structural Encoder.
    
    The Edge Predictor enables the "Causal" in Causal Binding Mechanism
    by predicting relationships between structural slots.
    
    Reference: Based on relational reasoning in Santoro et al. (2017).
    """
    num_heads: int = 4
    dropout: float = 0.1
    temperature: float = 1.0  # Lower = sharper edge weights


@dataclass
class StructuralEncoderConfig:
    """Configuration for Structural Encoder."""
    enabled: bool = True
    num_slots: int = 8
    num_layers: int = 12
    d_model: int = 512
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.15  # NeuroGen transfer: better OOD regularization
    abstraction_layer: AbstractionLayerConfig = field(default_factory=AbstractionLayerConfig)
    slot_attention: SlotAttentionConfig = field(default_factory=SlotAttentionConfig)
    # Edge Predictor for causal slot relationships
    use_edge_prediction: bool = True  # CRITICAL: Enables causal intervention in CBM
    edge_predictor: EdgePredictorConfig = field(default_factory=EdgePredictorConfig)


@dataclass
class ContentEncoderConfig:
    """Configuration for Content Encoder.
    
    NOTE: Lightweight (2 layers) since content is well-represented in pretrained embeddings.
    """
    enabled: bool = True
    num_layers: int = 2  # Lightweight - pretrained embeddings already capture content
    d_model: int = 512
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    pooling: str = "mean"  # mean or max
    use_orthogonal_projection: bool = False


@dataclass
class CausalBindingConfig:
    """Configuration for Causal Binding Mechanism.
    
    For TinyLlama (22 layers): inject at ~27%, ~50%, ~73% depth.
    """
    enabled: bool = True
    d_model: int = 2048  # TinyLlama hidden size
    # For TinyLlama (22 layers): inject at early (~27%), mid (~50%), late (~73%) layers
    injection_layers: List[int] = field(default_factory=lambda: [6, 11, 16])
    num_heads: int = 8
    dropout: float = 0.1
    use_causal_intervention: bool = True
    injection_method: str = "gated"  # gated or additive
    gate_init: float = 0.1
    use_structural_eos: bool = True  # Use structural slot coverage to predict EOS
    use_rope_broadcast: bool = True  # Use RoPE for length generalization (not learned position embeddings)


@dataclass
class PositionalEncodingConfig:
    """Configuration for positional encoding."""
    type: str = "rotary"  # rotary or alibi
    max_length: int = 1024  # Support prompt + long outputs (SCAN length split needs 288+ tokens)
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

    # CRITICAL #20: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "AdamW"
    lr: float = 2e-5
    base_lr: float = 2e-5      # Base model learning rate
    sci_lr: float = 5e-5       # SCI modules learning rate (2.5x higher)
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    max_grad_norm: float = 1.0  # Gradient clipping norm
    use_scheduler: bool = False  # Whether to use LR scheduler
    
    # CRITICAL #1: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "cosine"
    num_training_steps: int = 50000
    num_warmup_steps: int = 1000
    
    # CRITICAL #1: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    patience: int = 5  # Number of epochs to wait before stopping
    min_delta: float = 0.001  # Minimum change to qualify as improvement
    overfitting_threshold: float = 1.5  # Train/val loss ratio threshold
    
    # CRITICAL #1: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    max_epochs: int = 50
    epochs: int = 50  # Alias for backwards compatibility with train.py
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    mixed_precision: bool = True  # Use fp16
    eval_freq: int = 1  # Evaluate every N epochs
    save_every: int = 5  # Save checkpoint every N epochs
    gradient_accumulation_steps: int = 1  # For larger effective batch sizes

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    # CRITICAL #20: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class LossConfig:
    """Loss function weights and parameters.
    
    NeuroGen Transfer: Sharper temperature, stronger SCL weight, more EOS emphasis.
    """
    task_weight: float = 1.0
    scl_weight: float = 0.5  # NeuroGen: stronger structural learning signal
    scl_warmup_steps: int = 5000
    scl_warmup_epochs: int = 3  # NeuroGen: longer warmup for stability
    ortho_weight: float = 0.1
    eos_weight: float = 3.0  # NeuroGen: reliable sequence termination
    scl_temperature: float = 0.05  # NeuroGen: sharper structural discrimination

    # CRITICAL #20: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "scan"
    split: str = "length"  # or "template", "addprim_jump"
    max_length: int = 512  # Must be >= 300 for SCAN length split outputs (up to 288 tokens)
    scl_ratio: float = 0.5  # 50% of batch for SCL pairs
    num_workers: int = 4
    pairs_cache_dir: str = ".cache/scan"  # Cache for pair matrices
    force_regenerate_pairs: bool = False  # Force regeneration of pairs
    use_chat_template: bool = False  # Use chat template for chat-finetuned models like TinyLlama-Chat

    # CRITICAL #20: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    batch_size: int = 64
    beam_size: int = 1  # Greedy decoding for fair comparison
    max_generation_length: int = 512
    num_beams: int = 1           # Number of beams for generation
    do_sample: bool = False       # Greedy decoding (no sampling)
    repetition_penalty: float = 1.0  # No penalty (let model learn naturally)
    length_penalty: float = 1.0      # No penalty
    compute_structural_invariance: bool = False  # Optional: compute structural invariance metric
    # Default evaluation datasets (used by evaluate.py)
    datasets: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'name': 'scan', 'split': 'length', 'subset': 'test'}
    ])
    
    # CRITICAL #20: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    wandb_project: str = "SCI-SCAN"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    log_every_n_steps: int = 100
    log_every: int = 10  # Alias for trainer compatibility (log every N batches)
    checkpoint_every_n_epochs: int = 5
    use_wandb: bool = False  # Default off for local runs
    log_dir: str = "logs"
    results_dir: str = "results"
    
    # CRITICAL #20: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class CheckpointingConfig:
    """Checkpointing configuration."""
    save_dir: str = "checkpoints"
    keep_last_n: int = 3
    save_total_limit: int = 3  # Alias used by CheckpointManager
    
    # CRITICAL #1: Dict-style access support
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class ExpectedResultsConfig:
    """Expected results for validation (optional, used by evaluate.py)."""
    # Standard split names
    length: float = 0.0  # Expected exact match on length split
    simple: float = 0.0  # Expected exact match on simple split
    template: float = 0.0  # Expected exact match on template split
    addprim_jump: float = 0.0  # Expected exact match on addprim_jump split
    addprim_turn_left: float = 0.0  # Expected exact match on addprim_turn_left split
    # Alternative naming used in configs
    scan_length_id: float = 0.0  # In-distribution on length split
    scan_length_ood: float = 0.0  # Out-of-distribution on length split
    scan_simple: float = 0.0  # Simple split accuracy
    structural_invariance: float = 0.0  # Structural invariance metric


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
    expected_results: Optional[ExpectedResultsConfig] = None  # Optional: for validation

    # Random seeds for reproducibility
    seed: int = 42

    # CRITICAL #20: Add dict-style access support for backward compatibility
    def __getitem__(self, key):
        """Support dict-style access: config['key']"""
        return getattr(self, key)

    def get(self, key, default=None):
        """Support dict.get() style access"""
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)


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

    # HIGH #33: Validate config values
    _validate_config(config)

    return config


def _validate_config(config: SCIConfig):
    """
    Validate configuration values.

    HIGH #33: Ensure all config values are valid before training.
    """
    # Training validation
    assert config.training.batch_size > 0, \
        f"batch_size must be positive, got {config.training.batch_size}"
    assert config.training.max_epochs > 0, \
        f"max_epochs must be positive, got {config.training.max_epochs}"

    # Optimizer validation - handle both lr formats and types
    if hasattr(config.training.optimizer, 'base_lr'):
        lr_val = float(config.training.optimizer.base_lr) if isinstance(config.training.optimizer.base_lr, str) else config.training.optimizer.base_lr
        assert lr_val > 0, f"base_lr must be positive, got {config.training.optimizer.base_lr}"
    if hasattr(config.training.optimizer, 'sci_lr'):
        lr_val = float(config.training.optimizer.sci_lr) if isinstance(config.training.optimizer.sci_lr, str) else config.training.optimizer.sci_lr
        assert lr_val > 0, f"sci_lr must be positive, got {config.training.optimizer.sci_lr}"
    if hasattr(config.training.optimizer, 'lr'):
        lr_val = float(config.training.optimizer.lr) if isinstance(config.training.optimizer.lr, str) else config.training.optimizer.lr
        assert lr_val > 0, f"lr must be positive, got {config.training.optimizer.lr}"

    assert 0 <= config.training.optimizer.weight_decay < 1, \
        f"weight_decay must be in [0, 1), got {config.training.optimizer.weight_decay}"

    # Loss validation - allow 0 for baseline (no SCL)
    assert 0 <= config.loss.scl_weight <= 1, \
        f"scl_weight must be in [0, 1], got {config.loss.scl_weight}"
    assert config.loss.scl_temperature > 0, \
        f"scl_temperature must be positive, got {config.loss.scl_temperature}"
    assert 0 <= config.loss.ortho_weight <= 1, \
        f"ortho_weight must be in [0, 1], got {config.loss.ortho_weight}"

    # Data validation - CRITICAL: max_length for SCAN length split
    if config.data.split == "length":
        assert config.data.max_length >= 300, \
            f"SCAN length split requires max_length >= 300 (outputs up to 288 tokens), got {config.data.max_length}"
    
    # Evaluation max_generation_length should match data constraints
    if hasattr(config.evaluation, 'max_generation_length'):
        if config.data.split == "length":
            assert config.evaluation.max_generation_length >= 288, \
                f"SCAN length split requires max_generation_length >= 288, got {config.evaluation.max_generation_length}"

    # Model validation - only validate if enabled
    if getattr(config.model.structural_encoder, 'enabled', True):
        assert config.model.structural_encoder.num_slots > 0, \
            f"num_slots must be positive, got {config.model.structural_encoder.num_slots}"
        assert config.model.structural_encoder.d_model > 0, \
            f"d_model must be positive, got {config.model.structural_encoder.d_model}"

    # CBM validation - handle both attribute name formats
    cbm_enabled = getattr(config.model.causal_binding, 'enable_causal_intervention',
                          getattr(config.model.causal_binding, 'enabled', False))
    if cbm_enabled:
        assert len(config.model.causal_binding.injection_layers) > 0, \
            "injection_layers cannot be empty when causal intervention is enabled"
        assert all(layer >= 0 for layer in config.model.causal_binding.injection_layers), \
            f"injection_layers must be non-negative, got {config.model.causal_binding.injection_layers}"
        # MEDIUM #22: Validate injection_layers against num_decoder_layers
        num_decoder_layers = config.model.num_decoder_layers
        for layer in config.model.causal_binding.injection_layers:
            assert layer < num_decoder_layers, \
                f"injection_layer {layer} >= num_decoder_layers {num_decoder_layers}. " \
                f"Layers must be in range [0, {num_decoder_layers - 1}]"

    logger.info("✓ Config validation passed")


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
    
    # Normalize field names for backwards compatibility
    if 'model' in config_dict and isinstance(config_dict['model'], dict):
        model_dict = config_dict['model']
        # Handle base_model -> base_model_name
        if 'base_model' in model_dict and 'base_model_name' not in model_dict:
            model_dict['base_model_name'] = model_dict.pop('base_model')
    
    # Helper function to convert nested dicts
    def convert_section(section_dict: Dict[str, Any], section_class):
        if section_dict is None:
            return section_class()

        kwargs = {}
        for key, value in section_dict.items():
            # Skip unknown keys (for forward compatibility)
            if key not in section_class.__dataclass_fields__:
                logger.warning(f"Ignoring unknown config field: {key}")
                continue
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

    # Handle expected_results if present (optional)
    if 'expected_results' in config_dict:
        config_kwargs['expected_results'] = convert_section(
            config_dict['expected_results'], ExpectedResultsConfig
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
