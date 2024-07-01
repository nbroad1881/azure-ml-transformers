from typing import Any, Optional, Tuple
from dataclasses import dataclass

from transformers import DataCollatorForLanguageModeling

@dataclass
class DataCollatorForOnlyMaskingLanguageModeling(DataCollatorForLanguageModeling):
    """
    Typically, Masked Language Modeling operates on 15% of the tokens. Of those 15%, 80% are masked, 10% are random words, and 10% are the original words.
    This will make 100% of the specified masking probability to be masked tokens. This does NOT do random words or original words.
    (See this for more details: https://huggingface.co/papers/2202.08005)
    """


    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        labels[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels