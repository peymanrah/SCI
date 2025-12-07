"""
SCAN Data Collator with Proper EOS Handling

Required by: SCI_ENGINEERING_STANDARDS.md Section 4.3

LOW #77: Instruction Mask Format Documentation
----------------------------------------------
The instruction_mask is a binary tensor [batch, seq_len] where:
- 1 = instruction token (command input) - visible to structural encoder
- 0 = response token (action output) - masked from structural encoder

This prevents data leakage by ensuring the structural encoder only sees
the input command structure, not the target action sequence.

Example:
    Input:  "jump left"  -> [1, 1, 1]
    Output: "LTURN JUMP" -> [0, 0, 0]
    Full sequence: [1, 1, 1, 0, 0, 0]
"""

import torch


class SCANDataCollator:
    """Data collator with proper EOS handling for SCAN dataset."""

    def __init__(self, tokenizer, max_length=512):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (MEDIUM #69: Exposed as parameter)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # HIGH #27: Enforce padding_side='right' for decoder models
        # This is critical for proper attention mask handling
        self.tokenizer.padding_side = 'right'

        # Ensure EOS token is set
        assert tokenizer.eos_token_id is not None, "EOS token not set!"

        # Ensure PAD token is set (set to EOS if not available)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features):
        """
        Collate batch of features.

        Args:
            features: List of dicts with 'commands' and 'actions' keys

        Returns:
            dict: Batch with input_ids, attention_mask, labels, instruction_mask, commands
        """
        # Extract inputs and targets
        inputs = [f['commands'] for f in features]
        targets = [f['actions'] for f in features]

        # Encode inputs
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Encode targets WITH EOS token
        target_encodings = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # CRITICAL #7: Consolidated EOS enforcement logic
        # Ensure EOS token is at the end of each target sequence
        labels = target_encodings['input_ids'].clone()

        for i in range(len(targets)):
            # Find the last non-padding position
            non_pad_mask = labels[i] != self.tokenizer.pad_token_id
            if non_pad_mask.any():
                last_non_pad_idx = non_pad_mask.nonzero(as_tuple=False)[-1].item()

                # Ensure last token is EOS (replace if not)
                if labels[i, last_non_pad_idx] != self.tokenizer.eos_token_id:
                    # Try to add EOS after current last token if space available
                    if last_non_pad_idx + 1 < labels.shape[1]:
                        labels[i, last_non_pad_idx + 1] = self.tokenizer.eos_token_id
                    else:
                        # No space, replace last token with EOS
                        labels[i, last_non_pad_idx] = self.tokenizer.eos_token_id

        # Mask padding in labels with -100 (ignored by loss)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Create instruction mask
        # Instruction mask is 1 for input tokens, 0 for response tokens
        # For SCAN: instruction = command, response = actions
        # During training, SE and CE should only see the command (instruction_mask=1)
        instruction_mask = self._create_instruction_mask(input_encodings)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
            'instruction_mask': instruction_mask,
            'commands': inputs  # CRITICAL #5: Include commands for pair generation
        }

    def _create_instruction_mask(self, inputs):
        """
        Create mask that is 1 for instruction tokens, 0 for response tokens.

        CRITICAL #6: This prevents data leakage by ensuring structural/content encoders
        only see the instruction, not the response.

        For SCAN with separate inputs/targets, the input IS the instruction.
        The instruction mask should be 1 for all input tokens.

        Args:
            inputs: Encoded inputs (commands only, not actions)

        Returns:
            torch.Tensor: Instruction mask [batch, seq_len]
        """
        # For SCAN with separate encoding, inputs contain ONLY commands
        # So instruction_mask = attention_mask (1 for command tokens, 0 for padding)
        # This is correct because actions are encoded separately as labels
        instruction_mask = inputs['attention_mask'].clone()

        return instruction_mask
