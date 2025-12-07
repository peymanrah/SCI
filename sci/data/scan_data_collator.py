"""
SCAN Data Collator with Proper Causal LM Format

Required by: SCI_ENGINEERING_STANDARDS.md Section 4.3

BUG #86, #87, #90 FIX: Unified Collator for Causal LM Training
---------------------------------------------------------------
For causal language model finetuning:
1. Concatenate instruction + response into a SINGLE sequence
2. Create labels with -100 for instruction tokens (loss only on response)
3. Create instruction_mask for SE/CE (only see instruction, not response)

Format for each example:
    Full sequence: [instruction_tokens] + [separator] + [response_tokens] + [EOS]
    Labels:        [-100, -100, ..., -100, response_token_1, ..., response_token_n, EOS]
    instruction_mask: [1, 1, ..., 1, 0, 0, ..., 0, 0]

This ensures:
- Model learns next-token prediction on the full sequence
- Loss is only computed on response tokens
- SE/CE only see instruction tokens (no data leakage)
"""

import torch
from typing import List, Dict, Optional


class SCANDataCollator:
    """
    Data collator for SCAN dataset with proper causal LM format.
    
    BUG #86, #87, #90 FIX: This collator:
    1. Concatenates instruction + response into single sequence
    2. Creates proper labels with -100 for instruction tokens
    3. Creates instruction_mask for SE/CE data leakage prevention
    4. Includes commands for pair generation
    5. Optionally accepts pair_generator for batch pair labels
    6. Supports chat template for chat-finetuned models (e.g., TinyLlama-Chat)
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        separator: str = " -> ",  # Separator between command and actions
        pair_generator=None,  # Optional: for SCL pair labels
        use_chat_template: bool = False,  # Use tokenizer's chat template if available
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            separator: String separator between instruction and response
            pair_generator: Optional SCANPairGenerator for pair labels
            use_chat_template: If True and tokenizer has chat_template, format as chat
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator = separator
        self.pair_generator = pair_generator
        
        # Determine if we should use chat template
        self.use_chat_template = use_chat_template and hasattr(tokenizer, 'apply_chat_template')
        if use_chat_template and not self.use_chat_template:
            import warnings
            warnings.warn("use_chat_template=True but tokenizer has no apply_chat_template method. Using plain format.")

        # HIGH #27: Enforce padding_side='right' for decoder models
        self.tokenizer.padding_side = 'right'

        # Ensure EOS token is set
        assert tokenizer.eos_token_id is not None, "EOS token not set!"

        # Ensure PAD token is set (set to EOS if not available)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token

    def _format_instruction(self, command: str) -> str:
        """
        Format the command as instruction, optionally using chat template.
        
        For chat models like TinyLlama-Chat, this wraps the command in the
        proper chat format for better model understanding.
        """
        if self.use_chat_template:
            # Format as chat message for chat-finetuned models
            messages = [{"role": "user", "content": f"Translate to action sequence: {command}"}]
            # apply_chat_template returns the formatted string
            formatted = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True  # Add the assistant prefix
            )
            return formatted
        else:
            # Plain format: just the command
            return command

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features into proper causal LM format.

        Args:
            features: List of dicts with 'commands' and 'actions' keys
                     (and optionally 'idx' for pair lookup)

        Returns:
            dict: Batch with:
                - input_ids: [batch, seq_len] - Concatenated instruction + response
                - attention_mask: [batch, seq_len] - 1 for tokens, 0 for padding
                - labels: [batch, seq_len] - Same as input_ids but -100 for instruction
                - instruction_mask: [batch, seq_len] - 1 for instruction, 0 for response
                - commands: List[str] - Raw commands for pair generation
                - pair_labels: [batch, batch] - If pair_generator provided
        """
        # Extract inputs and targets
        commands = [f['commands'] for f in features]
        actions = [f['actions'] for f in features]
        indices = [f.get('idx', i) for i, f in enumerate(features)]

        batch_size = len(commands)

        # Tokenize each part separately, then combine for accurate boundary tracking
        all_input_ids = []
        all_instruction_lengths = []
        
        for cmd, act in zip(commands, actions):
            # Format instruction (optionally with chat template)
            instruction = self._format_instruction(cmd)
            
            # Tokenize instruction (with special tokens)
            inst_enc = self.tokenizer.encode(instruction, add_special_tokens=True)
            
            # Tokenize separator (no special tokens) - only if not using chat template
            # Chat template already includes proper formatting
            if self.use_chat_template:
                sep_enc = []  # Chat template already includes generation prompt
            else:
                sep_enc = self.tokenizer.encode(self.separator, add_special_tokens=False)
            
            # Tokenize response (no special tokens, but add EOS at end)
            act_enc = self.tokenizer.encode(act, add_special_tokens=False)
            
            # Instruction = formatted command + separator (everything before response)
            instruction_length = len(inst_enc) + len(sep_enc)
            
            # Full sequence = instruction + separator + response + EOS
            full_sequence = inst_enc + sep_enc + act_enc + [self.tokenizer.eos_token_id]
            
            all_input_ids.append(full_sequence)
            all_instruction_lengths.append(instruction_length)

        # Pad sequences to same length
        max_len = min(max(len(seq) for seq in all_input_ids), self.max_length)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for seq in all_input_ids:
            if len(seq) > max_len:
                # Truncate
                seq = seq[:max_len-1] + [self.tokenizer.eos_token_id]
            
            # Pad to max_len
            pad_len = max_len - len(seq)
            padded_seq = seq + [self.tokenizer.pad_token_id] * pad_len
            attention = [1] * len(seq) + [0] * pad_len
            
            padded_input_ids.append(padded_seq)
            padded_attention_mask.append(attention)

        input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)
        seq_len = input_ids.shape[1]

        # Create labels: same as input_ids but with -100 for instruction tokens and padding
        labels = input_ids.clone()
        
        # Create instruction_mask: 1 for instruction tokens, 0 for response/padding
        instruction_mask = torch.zeros_like(input_ids)

        for i in range(batch_size):
            inst_len = min(all_instruction_lengths[i], seq_len)
            
            # instruction_mask = 1 for instruction tokens, 0 for response
            instruction_mask[i, :inst_len] = 1
            
            # labels = -100 for instruction tokens (no loss computed)
            labels[i, :inst_len] = -100

        # Mask padding in labels with -100
        labels[attention_mask == 0] = -100

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'instruction_mask': instruction_mask,
            'commands': commands,  # CRITICAL #5: Include commands for pair generation
        }

        # Add pair labels if pair_generator is available
        if self.pair_generator is not None:
            pair_labels = self.pair_generator.get_batch_pair_labels(commands)
            result['pair_labels'] = pair_labels
        elif len(indices) > 0 and hasattr(features[0], 'pair_matrix'):
            # Fallback: try to get from dataset if available
            pass  # Will be handled by training loop

        return result

    def _create_instruction_mask(self, input_ids: torch.Tensor, instruction_lengths: List[int]) -> torch.Tensor:
        """
        Create mask that is 1 for instruction tokens, 0 for response tokens.

        CRITICAL #6: This prevents data leakage by ensuring structural/content encoders
        only see the instruction, not the response.

        Args:
            input_ids: [batch, seq_len] - Tokenized sequences
            instruction_lengths: List of instruction lengths for each example

        Returns:
            torch.Tensor: Instruction mask [batch, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        instruction_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)

        for i, inst_len in enumerate(instruction_lengths):
            instruction_mask[i, :inst_len] = 1

        return instruction_mask
