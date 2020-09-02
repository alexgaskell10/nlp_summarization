import logging
from typing import List, Optional, Tuple, Union
import torch
torch.manual_seed(0)

from transformers.tokenization_bart import BartTokenizer
from transformers.tokenization_utils_base import (
    ExplicitEnum, TruncationStrategy, PaddingStrategy, BatchEncoding,

)

logger = logging.getLogger(__name__)

class CustomTruncationStrategy(ExplicitEnum):
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"
    RANDOM_START = "random_start"

    def __init__(self, arg):
        pass

class CustomBartTokenizer(BartTokenizer):
    ''' Inherits from transformers.tokenization_utils.PreTrainedTokenizer.
        Allows selecting x adjacent tokens from random starting point in sequence
        (rather than only taking the first x tokens)
    '''
    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "only_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """ Truncates a sequence pair in place to the maximum length.

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to ``0``):
                number of tokens to remove using the truncation strategy
            truncation_strategy (:obj:`string`, `optional`, defaults to "only_first"):
                String selected in the following options:

                - 'only_first' (default): Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'longest_first': Iteratively reduce the inputs sequence until the input is under max_length
                  starting from the longest one at each token (when there is a pair of input sequences).
                  Overflowing tokens only contains overflow from the first sequence.
                - 'do_not_truncate'
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
        """
        overflowing_tokens = []
        if truncation_strategy == CustomTruncationStrategy.RANDOM_START:
            seq_len = len(ids) - num_tokens_to_remove
            assert seq_len > 0

            if num_tokens_to_remove <= 0:
                self.log_random_start_info(seq_len, num_tokens_to_remove, 0)
            else:
                starting_position = torch.randint(num_tokens_to_remove, (1,)).item()
                overflowing_tokens = ids[:starting_position] + ids[starting_position + seq_len:]
                ids = ids[starting_position: starting_position + seq_len]
                self.log_random_start_info(seq_len, num_tokens_to_remove, starting_position)

            return (ids, pair_ids, overflowing_tokens)

        elif num_tokens_to_remove <= 0:
            return (ids, pair_ids, [])
        
        else:
            return super().truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=num_tokens_to_remove,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=False, max_length=None, verbose=True, **kwargs
    ):
        """ Find the correct padding/truncation strategy with backward compatibility
            for old arguments (truncation_strategy and pad_to_max_length) and behaviors.
        """
        if (kwargs and 'truncation_strategy' not in kwargs and not \
                kwargs['truncation_strategy'] == CustomTruncationStrategy.RANDOM_START) or \
                (kwargs and not kwargs['truncation_strategy']):
            return super()._get_padding_truncation_strategies(
               padding, truncation, max_length, verbose, **kwargs
            )
        else:
            padding_strategy, truncation_strategy, max_length, kwargs = super()._get_padding_truncation_strategies(
                padding, truncation, max_length, verbose, **kwargs
            )
            truncation_strategy = CustomTruncationStrategy.RANDOM_START

            return padding_strategy, truncation_strategy, max_length, kwargs

    def log_random_start_info(
        self, tokenized_seq_len, num_tokens_to_remove, starting_position,
    ):
        ''' Helper to log info to reconstruct the random start data '''
        
        if not hasattr(self, 'random_start_info'):
            self.random_start_info = {
                'tokenized_seq_len': [],
                'num_tokens_to_remove': [],
                'starting_position': [],
            }
        
        self.random_start_info['tokenized_seq_len'].append(tokenized_seq_len) 
        self.random_start_info['num_tokens_to_remove'].append(num_tokens_to_remove)
        self.random_start_info['starting_position'].append(starting_position)

    def save_random_start_info(self, outfile):
        ''' Helper to save random start info to file upon finishing tokenization '''
        import json
        with open(outfile, 'w') as f:
            json.dump(self.random_start_info, f)
        
        self.random_start_info = {
            'tokenized_seq_len': [],
            'num_tokens_to_remove': [],
            'starting_position': [],
        }

    def _prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[str] = None,
        prepend_batch_axis: bool = False,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """ Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        if max_length is None and len(encoded_inputs["input_ids"]) > self.model_max_length and verbose:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(ids), self.model_max_length)
            )

        # Padding
        encoded_inputs = self.pad(
            encoded_inputs,
            max_length=max_length,
            padding=padding_strategy.value,
            return_attention_mask=return_attention_mask,
        )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs
