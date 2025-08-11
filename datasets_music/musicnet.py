import torch
import os
import datasets
import torch.utils.data as data
import bd3lms.utils as bd_utils

# Get empty tokenizer for special tokens
empty_tokenizer = bd_utils.EmptyTokenizer()
bos_token = empty_tokenizer.bos_token_id
eos_token = empty_tokenizer.eos_token_id
pad_token = empty_tokenizer.pad_token_id

import json
from typing import Dict, List

    
class TokenizedMTG2(data.Dataset):
    """Dataset for loading pre-computed MTG tokens in a format that is
    compatible with the `bd3lms` dataloader utility which expects a
    🤗 `datasets.Dataset`-like interface (supports `.map`, `.remove_columns`,
    `.with_format`, etc.).

    Each `.pt` file under ``root`` is assumed to contain a 1-D ``torch.LongTensor``
    with the encoded tokens of a single MTG example.
    """

    def __init__(self, root: str = "/home/tzlillev/LLadaSMDM/MTG_Tokens"):
        self.root = root
        self.tag_converter = bd_utils.TagsTokenizer()

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Provided root directory '{self.root}' does not exist.")

        # Discover all token files
        self.files = sorted([f for f in os.listdir(self.root) if f.endswith(".pt")])
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt token files found under '{self.root}'.")

        # Randomly select 50% of files
        num_files = len(self.files)
        indices = torch.randperm(num_files)[:num_files//2]
        self.files = [self.files[i] for i in indices]
        print(f"Randomly selected {len(self.files)} files out of {num_files} total files")

        # Use generator-based dataset creation
        self._dataset = datasets.Dataset.from_generator(self._generate_examples)

    def _generate_examples(self):
        """Yields examples from the token files."""
        for fname in self.files:
            path = os.path.join(self.root, fname)
         #   print(f"Loading file {fname}")
            # Extract just the number from filenames like 1000082_encodec_CB4.pt
            file_id = int(fname.split('_')[0])
            track_tokens = self.tag_converter.tokenize(file_id)
            if "error" in track_tokens:
                print(f"Skipping track {file_id} due to: {track_tokens['error']}")
                continue
          #  print(f"Track tokens: {track_tokens} for file id {file_id}")
            # Load tensor to CPU explicitly to avoid CUDA errors
            tokens = torch.load(path)
            
            # Ensure we end up with a 1-D LongTensor regardless of the on-disk
            # structure.
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.long().view(-1)
            elif isinstance(tokens, list):
                # Some files store a list of tensors/chunks. Concatenate.
                if all(isinstance(t, torch.Tensor) for t in tokens):
                    tokens = torch.cat([t.view(-1).long() for t in tokens], dim=0)
                else:
                    # List of ints/other – convert directly
                    tokens = torch.tensor(tokens, dtype=torch.long).view(-1)
            else:
                # Unknown structure, try best-effort conversion
                tokens = torch.as_tensor(tokens, dtype=torch.long).view(-1)
            
            # Add BOS tokens at start and EOS tokens at end
            tokens_list = tokens.tolist()
            tokens_list = [bos_token] * 4 + tokens_list + [eos_token] * 4
            
            yield {"input_ids": tokens_list, "text": "", "track_tokens": track_tokens["tokens"]}

    # ---------------------------------------------------------------------
    # Basic torch.utils.data.Dataset protocol (len / getitem)
    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        """Return a single example.
        With robust error handling for CUDA issues.
        """
        if isinstance(idx, str):
            if idx in {"train", "valid", "validation", "test"}:
                return self  # Return the proxy dataset itself for split indexing
            raise ValueError(f"Unknown split key '{idx}'. Expected 'train', 'valid', or 'test'.")
        try:
            item = self._dataset[idx]
            return torch.tensor(item["input_ids"], dtype=torch.bfloat16)
        except Exception as e:
            print(f"Error retrieving item {idx}: {e}")
            raise ValueError(f"Error retrieving item {idx}: {e} - TZLIL WAS HERE")
            # Return a default small tensor as fallback
            return torch.zeros(1, dtype=torch.long)

    # ---------------------------------------------------------------------
    # Delegate common dataset ops to the internal HuggingFace dataset
    # ---------------------------------------------------------------------
    def __getattr__(self, name):
        """Forward unknown attributes/methods to the underlying HF dataset."""
        hf_ds = super().__getattribute__("_dataset")
        if hasattr(hf_ds, name):
            attr = getattr(hf_ds, name)
            # If the attribute is a method that returns another HF dataset we
            # want to keep the chaining behaviour working. Wrap the returned
            # dataset back into our proxy class when appropriate.
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return result  # usually another Dataset, which is fine
                return wrapper
            return attr
        raise AttributeError(name)

    # Explicit wrappers for clarity (these just call through to the HF dataset)
    def map(self, function=None, *args, **kwargs):
        # If the upstream pipeline tries to apply its own tokenizer we can
        # safely skip that step because we already provide tokenized inputs.
        if function is not None and getattr(function, "__name__", "") == "preprocess_and_tokenize":
            # No-op – return the proxy dataset unchanged
            return self

        return self._dataset.map(function, *args, **kwargs)

    def remove_columns(self, *args, **kwargs):
        # Gracefully handle attempts to remove non-existent columns (e.g. 'text').
        columns = args[0] if args else kwargs.get("column_names", [])
        if isinstance(columns, str):
            columns = [columns]

        existing = [c for c in columns if c in self._dataset.column_names]
        if len(existing) == 0:
            return self._dataset  # nothing to remove

        return self._dataset.remove_columns(existing)

    def with_format(self, *args, **kwargs):
        return self._dataset.with_format(*args, **kwargs)

    def save_to_disk(self, *args, **kwargs):
        return self._dataset.save_to_disk(*args, **kwargs)
    




class TokenizedMTG(data.Dataset):
    """Dataset for loading pre-computed MTG tokens in a format that is
    compatible with the `bd3lms` dataloader utility which expects a
    🤗 `datasets.Dataset`-like interface (supports `.map`, `.remove_columns`,
    `.with_format`, etc.).

    Each `.pt` file under ``root`` is assumed to contain a 1-D ``torch.LongTensor``
    with the encoded tokens of a single MTG example.
    """

    def __init__(self, root: str = "./MTG_Tokens"):
        self.root = root
        self.tag_converter = bd_utils.TagsTokenizer()

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Provided root directory '{self.root}' does not exist.")

        # Discover all token files
        self.files = sorted([f for f in os.listdir(self.root) if f.endswith(".pt")])
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt token files found under '{self.root}'.")

        # Load every token file once and keep as plain Python lists
        token_sequences = []
        track_tokens = []
        
        for fname in self.files:
            path = os.path.join(self.root, fname)
            #print(f"Loading file {fname}")
            # Extract just the number from filenames like 1000082_encodec_CB4.pt
            file_id = int(fname.split('_')[0])
            track_token = self.tag_converter.tokenize(file_id)
            if "error" in track_token:
                print(f"Skipping track {file_id} due to: {track_token['error']}")
                continue
            # Load tensor to CPU explicitly to avoid CUDA errors
            tokens = torch.load(path)
            track_tokens.append(track_token)
            # Ensure we end up with a 1-D LongTensor regardless of the on-disk
            # structure.
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.long().view(-1)
            elif isinstance(tokens, list):
                # Some files store a list of tensors/chunks. Concatenate.
                if all(isinstance(t, torch.Tensor) for t in tokens):
                    tokens = torch.cat([t.view(-1).long() for t in tokens], dim=0)
                else:
                    # List of ints/other – convert directly
                    tokens = torch.tensor(tokens, dtype=torch.long).view(-1)
            else:
                # Unknown structure, try best-effort conversion
                tokens = torch.as_tensor(tokens, dtype=torch.long).view(-1)
            
            # Add BOS tokens at start and EOS tokens at end
            tokens_list = tokens.tolist()
            tokens_list = [bos_token] * 4 + tokens_list + [eos_token] * 4
            
            token_sequences.append(tokens_list)

        # Build an internal HuggingFace dataset so we get `.map`, `.remove_columns`, etc.
        self._dataset = datasets.Dataset.from_dict({
            "input_ids": token_sequences, 
            "text": [""] * len(token_sequences), 
            "track_tokens": track_tokens
        })

    # ---------------------------------------------------------------------
    # Basic torch.utils.data.Dataset protocol (len / getitem)
    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        """Return a single example.
        With robust error handling for CUDA issues.
        """
        if isinstance(idx, str):
            if idx in {"train", "valid", "validation", "test"}:
                return self  # Return the proxy dataset itself for split indexing
            raise ValueError(f"Unknown split key '{idx}'. Expected 'train', 'valid', or 'test'.")
        try:
            item = self._dataset[idx]
            return torch.tensor(item["input_ids"], dtype=torch.bfloat16)
        except Exception as e:
            print(f"Error retrieving item {idx}: {e}")
            raise ValueError(f"Error retrieving item {idx}: {e} - TZLIL WAS HERE")
            # Return a default small tensor as fallback
            return torch.zeros(1, dtype=torch.long)

    # ---------------------------------------------------------------------
    # Delegate common dataset ops to the internal HuggingFace dataset
    # ---------------------------------------------------------------------
    def __getattr__(self, name):
        """Forward unknown attributes/methods to the underlying HF dataset."""
        hf_ds = super().__getattribute__("_dataset")
        if hasattr(hf_ds, name):
            attr = getattr(hf_ds, name)
            # If the attribute is a method that returns another HF dataset we
            # want to keep the chaining behaviour working. Wrap the returned
            # dataset back into our proxy class when appropriate.
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return result  # usually another Dataset, which is fine
                return wrapper
            return attr
        raise AttributeError(name)

    # Explicit wrappers for clarity (these just call through to the HF dataset)
    def map(self, function=None, *args, **kwargs):
        # If the upstream pipeline tries to apply its own tokenizer we can
        # safely skip that step because we already provide tokenized inputs.
        if function is not None and getattr(function, "__name__", "") == "preprocess_and_tokenize":
            # No-op – return the proxy dataset unchanged
            return self

        return self._dataset.map(function, *args, **kwargs)

    def remove_columns(self, *args, **kwargs):
        # Gracefully handle attempts to remove non-existent columns (e.g. 'text').
        columns = args[0] if args else kwargs.get("column_names", [])
        if isinstance(columns, str):
            columns = [columns]

        existing = [c for c in columns if c in self._dataset.column_names]
        if len(existing) == 0:
            return self._dataset  # nothing to remove

        return self._dataset.remove_columns(existing)

    def with_format(self, *args, **kwargs):
        return self._dataset.with_format(*args, **kwargs)

    def save_to_disk(self, *args, **kwargs):
        return self._dataset.save_to_disk(*args, **kwargs)


    


class TokenizedMusicNet_32khz_V2(data.Dataset):
    """Dataset for loading pre-computed MusicNet tokens in a format that is
    compatible with the `bd3lms` dataloader utility which expects a
    🤗 `datasets.Dataset`-like interface (supports `.map`, `.remove_columns`,
    `.with_format`, etc.).

    Each `.pt` file under ``root`` is assumed to contain a 1-D ``torch.LongTensor``
    with the encoded tokens of a single MusicNet example.
    """

    def __init__(self, root: str = "./MusicNet_32khz_Tokens"):
        self.root = root

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Provided root directory '{self.root}' does not exist.")

        # Discover all token files
        self.files = sorted([f for f in os.listdir(self.root) if f.endswith(".pt")])
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt token files found under '{self.root}'.")

        # Load every token file once and keep as plain Python lists so that we
        # can build a HuggingFace `Dataset`. (If memory becomes a concern you
        # can switch to a generator-based approach, but for the typical size of
        # MusicNet this is fine.)
        self._dataset = datasets.Dataset.from_generator(self._generate_examples)

    def _generate_examples(self):
        """Yields examples from the token files."""
        for idx, fname in enumerate(self.files):
            path = os.path.join(self.root, fname)
            tokens = torch.load(path)
            # Ensure we end up with a 1-D LongTensor regardless of the on-disk
            # structure.
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.long().view(-1)
            elif isinstance(tokens, list):
                # Some files store a list of tensors/chunks. Concatenate.
                if all(isinstance(t, torch.Tensor) for t in tokens):
                    tokens = torch.cat([t.view(-1).long() for t in tokens], dim=0)
                else:
                    # List of ints/other – convert directly
                    tokens = torch.tensor(tokens, dtype=torch.long).view(-1)
            else:
                # Unknown structure, try best-effort conversion
                tokens = torch.as_tensor(tokens, dtype=torch.long).view(-1)
            
            # Add BOS/EOS tokens
            tokens_list = tokens.tolist()
            tokens_list = [bos_token] * 4 + tokens_list + [eos_token] * 4
            yield {"input_ids": tokens_list, "text": ""}

    # ---------------------------------------------------------------------
    # Basic torch.utils.data.Dataset protocol (len / getitem)
    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        """Return a single example.
        With robust error handling for CUDA issues.
        """
        if isinstance(idx, str):
            if idx in {"train", "valid", "validation", "test"}:
                return self  # Return the proxy dataset itself for split indexing
            raise ValueError(f"Unknown split key '{idx}'. Expected 'train', 'valid', or 'test'.")
        try:
            item = self._dataset[idx]
            return torch.tensor(item["input_ids"], dtype=torch.bfloat16)
        except Exception as e:
            print(f"Error retrieving item {idx}: {e}")
            raise ValueError(f"Error retrieving item {idx}: {e} - TZLIL WAS HERE")
            # Return a default small tensor as fallback
            return torch.zeros(1, dtype=torch.long)

    # ---------------------------------------------------------------------
    # Delegate common dataset ops to the internal HuggingFace dataset
    # ---------------------------------------------------------------------
    def __getattr__(self, name):
        """Forward unknown attributes/methods to the underlying HF dataset."""
        hf_ds = super().__getattribute__("_dataset")
        if hasattr(hf_ds, name):
            attr = getattr(hf_ds, name)
            # If the attribute is a method that returns another HF dataset we
            # want to keep the chaining behaviour working. Wrap the returned
            # dataset back into our proxy class when appropriate.
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return result  # usually another Dataset, which is fine
                return wrapper
            return attr
        raise AttributeError(name)

    # Explicit wrappers for clarity (these just call through to the HF dataset)
    def map(self, function=None, *args, **kwargs):
        # If the upstream pipeline tries to apply its own tokenizer we can
        # safely skip that step because we already provide tokenized inputs.
        if function is not None and getattr(function, "__name__", "") == "preprocess_and_tokenize":
            # No-op – return the proxy dataset unchanged
            return self

        return self._dataset.map(function, *args, **kwargs)

    def remove_columns(self, *args, **kwargs):
        # Gracefully handle attempts to remove non-existent columns (e.g. 'text').
        columns = args[0] if args else kwargs.get("column_names", [])
        if isinstance(columns, str):
            columns = [columns]

        existing = [c for c in columns if c in self._dataset.column_names]
        if len(existing) == 0:
            return self._dataset  # nothing to remove

        return self._dataset.remove_columns(existing)

    def with_format(self, *args, **kwargs):
        return self._dataset.with_format(*args, **kwargs)

    def save_to_disk(self, *args, **kwargs):
        return self._dataset.save_to_disk(*args, **kwargs)
    
class TokenizedMusicNet_32khz(data.Dataset):
    """Dataset for loading pre-computed MusicNet tokens in a format that is
    compatible with the `bd3lms` dataloader utility which expects a
    🤗 `datasets.Dataset`-like interface (supports `.map`, `.remove_columns`,
    `.with_format`, etc.).

    Each `.pt` file under ``root`` is assumed to contain a 1-D ``torch.LongTensor``
    with the encoded tokens of a single MusicNet example.
    """

    def __init__(self, root: str = "./MusicNet_32khz_Tokens"):
        self.root = root

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Provided root directory '{self.root}' does not exist.")

        # Discover all token files
        self.files = sorted([f for f in os.listdir(self.root) if f.endswith(".pt")])
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt token files found under '{self.root}'.")

        # Load every token file once and keep as plain Python lists so that we
        # can build a HuggingFace `Dataset`. (If memory becomes a concern you
        # can switch to a generator-based approach, but for the typical size of
        # MusicNet this is fine.)
        token_sequences = []
        
        for fname in self.files:
            path = os.path.join(self.root, fname)
            # Load tensor to CPU explicitly to avoid CUDA errors
            tokens = torch.load(path)
            # Ensure we end up with a 1-D LongTensor regardless of the on-disk
            # structure.
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.long().view(-1)
            elif isinstance(tokens, list):
                # Some files store a list of tensors/chunks. Concatenate.
                if all(isinstance(t, torch.Tensor) for t in tokens):
                    tokens = torch.cat([t.view(-1).long() for t in tokens], dim=0)
                else:
                    # List of ints/other – convert directly
                    tokens = torch.tensor(tokens, dtype=torch.long).view(-1)
            else:
                # Unknown structure, try best-effort conversion
                tokens = torch.as_tensor(tokens, dtype=torch.long).view(-1)
            
            # Add BOS tokens at start and EOS tokens at end
            tokens_list = tokens.tolist()
            tokens_list = [bos_token] * 4 + tokens_list + [eos_token] * 4
            
            token_sequences.append(tokens_list)

        # Build an internal HuggingFace dataset so we get `.map`, `.remove_columns`, etc.
        self._dataset = datasets.Dataset.from_dict({"input_ids": token_sequences, "text": [""] * len(token_sequences)})

    # ---------------------------------------------------------------------
    # Basic torch.utils.data.Dataset protocol (len / getitem)
    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        """Return a single example.
        With robust error handling for CUDA issues.
        """
        if isinstance(idx, str):
            if idx in {"train", "valid", "validation", "test"}:
                return self  # Return the proxy dataset itself for split indexing
            raise ValueError(f"Unknown split key '{idx}'. Expected 'train', 'valid', or 'test'.")
        try:
            item = self._dataset[idx]
            return torch.tensor(item["input_ids"], dtype=torch.bfloat16)
        except Exception as e:
            print(f"Error retrieving item {idx}: {e}")
            raise ValueError(f"Error retrieving item {idx}: {e} - TZLIL WAS HERE")
            # Return a default small tensor as fallback
            return torch.zeros(1, dtype=torch.long)

    # ---------------------------------------------------------------------
    # Delegate common dataset ops to the internal HuggingFace dataset
    # ---------------------------------------------------------------------
    def __getattr__(self, name):
        """Forward unknown attributes/methods to the underlying HF dataset."""
        hf_ds = super().__getattribute__("_dataset")
        if hasattr(hf_ds, name):
            attr = getattr(hf_ds, name)
            # If the attribute is a method that returns another HF dataset we
            # want to keep the chaining behaviour working. Wrap the returned
            # dataset back into our proxy class when appropriate.
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return result  # usually another Dataset, which is fine
                return wrapper
            return attr
        raise AttributeError(name)

    # Explicit wrappers for clarity (these just call through to the HF dataset)
    def map(self, function=None, *args, **kwargs):
        # If the upstream pipeline tries to apply its own tokenizer we can
        # safely skip that step because we already provide tokenized inputs.
        if function is not None and getattr(function, "__name__", "") == "preprocess_and_tokenize":
            # No-op – return the proxy dataset unchanged
            return self

        return self._dataset.map(function, *args, **kwargs)

    def remove_columns(self, *args, **kwargs):
        # Gracefully handle attempts to remove non-existent columns (e.g. 'text').
        columns = args[0] if args else kwargs.get("column_names", [])
        if isinstance(columns, str):
            columns = [columns]

        existing = [c for c in columns if c in self._dataset.column_names]
        if len(existing) == 0:
            return self._dataset  # nothing to remove

        return self._dataset.remove_columns(existing)

    def with_format(self, *args, **kwargs):
        return self._dataset.with_format(*args, **kwargs)

    def save_to_disk(self, *args, **kwargs):
        return self._dataset.save_to_disk(*args, **kwargs)
    
class TokenizedMusicNet(data.Dataset):
    """Dataset for loading pre-computed MusicNet tokens in a format that is
    compatible with the `bd3lms` dataloader utility which expects a
    🤗 `datasets.Dataset`-like interface (supports `.map`, `.remove_columns`,
    `.with_format`, etc.).

    Each `.pt` file under ``root`` is assumed to contain a 1-D ``torch.LongTensor``
    with the encoded tokens of a single MusicNet example.
    """

    def __init__(self, root: str = "./musicnet/tokens/codebooks_9"):
        self.root = root

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Provided root directory '{self.root}' does not exist.")

        # Discover all token files
        self.files = sorted([f for f in os.listdir(self.root) if f.endswith(".pt")])
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt token files found under '{self.root}'.")

        # Load every token file once and keep as plain Python lists so that we
        # can build a HuggingFace `Dataset`. (If memory becomes a concern you
        # can switch to a generator-based approach, but for the typical size of
        # MusicNet this is fine.)
        token_sequences = []
        for fname in self.files:
            path = os.path.join(self.root, fname)
            # Load tensor to CPU explicitly to avoid CUDA errors
            tokens = torch.load(path)
            # Ensure we end up with a 1-D LongTensor regardless of the on-disk
            # structure.
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.long().view(-1)
            elif isinstance(tokens, list):
                # Some files store a list of tensors/chunks. Concatenate.
                if all(isinstance(t, torch.Tensor) for t in tokens):
                    tokens = torch.cat([t.view(-1).long() for t in tokens], dim=0)
                else:
                    # List of ints/other – convert directly
                    tokens = torch.tensor(tokens, dtype=torch.long).view(-1)
            else:
                # Unknown structure, try best-effort conversion
                tokens = torch.as_tensor(tokens, dtype=torch.long).view(-1)
            
            # Add BOS tokens at start and EOS tokens at end
            tokens_list = tokens.tolist()
            tokens_list = [bos_token] * 4 + tokens_list + [eos_token] * 4
            token_sequences.append(tokens_list)

        # Build an internal HuggingFace dataset so we get `.map`, `.remove_columns`, etc.
        self._dataset = datasets.Dataset.from_dict({"input_ids": token_sequences, "text": [""] * len(token_sequences)})

    # ---------------------------------------------------------------------
    # Basic torch.utils.data.Dataset protocol (len / getitem)
    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        """Return a single example.
        With robust error handling for CUDA issues.
        """
        if isinstance(idx, str):
            if idx in {"train", "valid", "validation", "test"}:
                return self  # Return the proxy dataset itself for split indexing
            raise ValueError(f"Unknown split key '{idx}'. Expected 'train', 'valid', or 'test'.")

        # Integer indexing – fetch from the internal HF dataset and cast to tensor
        try:
            item = self._dataset[idx]
            return torch.tensor(item["input_ids"], dtype=torch.long)
        except Exception as e:
            print(f"Error retrieving item {idx}: {e}")
            raise ValueError(f"Error retrieving item {idx}: {e} - TZLIL WAS HERE")
            # Return a default small tensor as fallback
            return torch.zeros(1, dtype=torch.long)

    # ---------------------------------------------------------------------
    # Delegate common dataset ops to the internal HuggingFace dataset
    # ---------------------------------------------------------------------
    def __getattr__(self, name):
        """Forward unknown attributes/methods to the underlying HF dataset."""
        hf_ds = super().__getattribute__("_dataset")
        if hasattr(hf_ds, name):
            attr = getattr(hf_ds, name)
            # If the attribute is a method that returns another HF dataset we
            # want to keep the chaining behaviour working. Wrap the returned
            # dataset back into our proxy class when appropriate.
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return result  # usually another Dataset, which is fine
                return wrapper
            return attr
        raise AttributeError(name)

    # Explicit wrappers for clarity (these just call through to the HF dataset)
    def map(self, function=None, *args, **kwargs):
        # If the upstream pipeline tries to apply its own tokenizer we can
        # safely skip that step because we already provide tokenized inputs.
        if function is not None and getattr(function, "__name__", "") == "preprocess_and_tokenize":
            # No-op – return the proxy dataset unchanged
            return self

        return self._dataset.map(function, *args, **kwargs)

    def remove_columns(self, *args, **kwargs):
        # Gracefully handle attempts to remove non-existent columns (e.g. 'text').
        columns = args[0] if args else kwargs.get("column_names", [])
        if isinstance(columns, str):
            columns = [columns]

        existing = [c for c in columns if c in self._dataset.column_names]
        if len(existing) == 0:
            return self._dataset  # nothing to remove

        return self._dataset.remove_columns(existing)

    def with_format(self, *args, **kwargs):
        return self._dataset.with_format(*args, **kwargs)

    def save_to_disk(self, *args, **kwargs):
        return self._dataset.save_to_disk(*args, **kwargs)

class TokenizedMusicNetMTG_32khz(data.Dataset):
    """Unified dataset that concatenates pre-tokenized MTG and 32 kHz MusicNet examples.

    This thin wrapper exposes the same HuggingFace-like interface used by the
    existing *Tokenized…* datasets in this file so the rest of the dataloader
    pipeline (e.g. bd3lms) can consume it without any code changes.

    Parameters
    ----------
    musicnet_root : str
        Folder that contains the 32 kHz MusicNet ``.pt`` token files.
    mtg_root : str
        Folder that contains the MTG ``.pt`` token files.
    add_source_column : bool, optional
        If ``True`` an additional string column named ``source`` will be added
        to indicate whether a sample originates from *MTG* or *MusicNet*.  This
        is useful for corpus-specific evaluation or sampling but **disabled by
        default** because downstream batching utilities that perform reshaping
        (e.g. `_group_texts` in *bd3lms*) usually drop unknown columns and will
        raise Arrow length-mismatch errors if the column is kept.
    """

    def __init__(self,
                 musicnet_root: str = "./MusicNet_32khz_Tokens",
                 mtg_root: str = "./MTG_Tokens",
                 add_source_column: bool = False):
        super().__init__()

        # Instantiate the two existing proxy datasets ----------------------
        mtg_ds = TokenizedMTG(mtg_root)
        musicnet_ds = TokenizedMusicNet_32khz(musicnet_root)

        # Collect their underlying HF datasets and concatenate -------------
        hf_mtg = mtg_ds._dataset
        hf_musicnet = musicnet_ds._dataset

        if add_source_column:
            # Add a simple provenance column *before* concatenation so that the
            # feature schemas match across the two datasets.
            hf_mtg = hf_mtg.add_column("source", ["MTG"] * len(hf_mtg))
            hf_musicnet = hf_musicnet.add_column("source", ["MusicNet"] * len(hf_musicnet))

        # Row-wise concatenation (keeps lazy memory-mapping behaviour).
        self._dataset = datasets.concatenate_datasets([hf_mtg, hf_musicnet])

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        """Return a single example.
        Mirrors the convention used by `TokenizedMTG`/`TokenizedMusicNet_32khz`:
        • string indices like "train", "valid", "test" are treated as split
          selectors and simply return the dataset itself (the bd3lms loader
          relies on this to obtain the same object back).
        • integer indices return the token sequence as a 1-D `torch.LongTensor`.
        """

        if isinstance(idx, str):
            if idx in {"train", "valid", "validation", "test"}:
                return self  # split indexing used by upstream code
            raise ValueError(
                f"Unknown split key '{idx}'. Expected 'train', 'valid', or 'test'."
            )

        try:
            item = self._dataset[idx]
            return torch.tensor(item["input_ids"], dtype=torch.long)
        except Exception as e:
            # Surface a clear error message but keep a fallback for robustness
            print(f"Error retrieving item {idx}: {e}")
            raise ValueError(f"Error retrieving item {idx}: {e} - TZLIL WAS HERE")
            return torch.zeros(1, dtype=torch.long)

    # ------------------------------------------------------------------
    # Delegate attribute/method access to the backing HF dataset so that
    # `.map`, `.remove_columns`, `.with_format`, … work transparently.
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        hf_ds = super().__getattribute__("_dataset")
        if hasattr(hf_ds, name):
            attr = getattr(hf_ds, name)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    return result
                return wrapper
            return attr
        raise AttributeError(name)

    # Explicit wrappers kept for readability/use-site parity --------------
    def map(self, function=None, *args, **kwargs):
        # Skip redundant tokenisation attempts coming from the upstream loader
        if function is not None and getattr(function, "__name__", "") == "preprocess_and_tokenize":
            return self  # already tokenised
        return self._dataset.map(function, *args, **kwargs)

    def remove_columns(self, *args, **kwargs):
        columns = args[0] if args else kwargs.get("column_names", [])
        if isinstance(columns, str):
            columns = [columns]
        existing = [c for c in columns if c in self._dataset.column_names]
        if len(existing) == 0:
            return self._dataset  # nothing to remove
        return self._dataset.remove_columns(existing)

    def with_format(self, *args, **kwargs):
        return self._dataset.with_format(*args, **kwargs)

    def save_to_disk(self, *args, **kwargs):
        return self._dataset.save_to_disk(*args, **kwargs)
    






# import csv
# import json
# from typing import List, Dict, Union
# import io
# import os
# import re
# import random

# # Define paths
# TSV_PATH = "/home/tzlillev/LLadaSMDM/mtg-jamendo-dataset/data/autotagging.tsv"
# COMBO_PATH = "/home/tzlillev/LLadaSMDM/mtg-jamendo-dataset/combo_dict_Fast.json"

# class TrackTagMapper:
#     """
#     A class to map track IDs and tags from the MTG-Jamendo dataset to a set of four tokens.
#     """

#     def __init__(self, tsv_content: str, combo_path: str = COMBO_PATH):
#         """
#         Initializes the TrackTagMapper.

#         Args:
#             tsv_content (str): The content of the autotagging.tsv file as a string.
#             combo_path (str): The path to the JSON file for storing the tag combination mapping.
#         """
#         self.tsv_content = tsv_content
#         self.combo_path = combo_path
#         # The order of these calls is important!
#         self._track_tags = self._load_track_tags()
#         self.combo_to_id = self._create_combo_to_id_mapping()
#         self.id_to_combo = {v: k for k, v in self.combo_to_id.items()}

#     def _load_track_tags(self) -> Dict[str, List[str]]:
#         """
#         ✅ FIXED: Loads all tags for each track from the TSV content.
#         This function now correctly handles files with a variable number of columns,
#         where all columns after the first 5 are considered tags.

#         Returns:
#             A dictionary mapping track IDs to a list of ALL its tags.
#         """
#         track_tags = {}
#         f = io.StringIO(self.tsv_content)
#         # Use csv.reader because the number of columns is not fixed.
#         reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
#         # Skip the header row
#         header = next(reader)
#         for row in reader:
#             if not row:  # Skip any empty rows
#                 continue
#             tid = row[0]
#             # All columns from the 5th index (6th column) onwards are tags.
#             tags = row[5:]
#             # Ensure no empty strings are included if there are trailing tabs.
#             track_tags[tid] = [tag for tag in tags if tag]
#         return track_tags

#     def _create_combo_to_id_mapping(self) -> Dict[str, int]:
#         """
#         Creates a mapping from unique tag combinations to an integer ID.
#         """
#         combo_to_id = {}
#         next_id = 0
#         for tags in self._track_tags.values():
#             key = self._tags_to_key(tags)
#             if key not in combo_to_id:
#                 combo_to_id[key] = next_id
#                 next_id += 1
#         return combo_to_id

#     def _tags_to_key(self, tags: List[str]) -> str:
#         """
#         Converts a list of tags into a standardized, sorted key.
#         """
#         genres = sorted([t.split('---')[1] for t in tags if t.startswith('genre---')])[:3]
#         moods = sorted([t.split('---')[1] for t in tags if t.startswith('mood/theme---')])[:3]
#         instruments = sorted([t.split('---')[1] for t in tags if t.startswith('instrument---')])[:3]

#         # Combine in a fixed order (genre, mood, instrument)
#         combo = genres + moods + instruments
#         combo = combo[:4]
#         # Pad with the last available tag or "NONE" if empty
#         while len(combo) < 4:
#             combo.append(combo[-1] if combo else "NONE")

#         return "|||".join(combo)

#     def _key_to_tokens(self, key: str) -> List[int]:
#         """
#         Converts a combo key into a list of four base-2048 tokens (starting from 1).
#         """
#         token_id = self.combo_to_id.get(key)
#         if token_id is None:
#             raise KeyError(f"Combo key '{key}' not found in the mapping. This specific combination of tags does not exist in the source dataset.")

#         tokens = []
#         for _ in range(4):
#             tokens.append((token_id % 2048) + 1)
#             token_id //= 2048
#         return tokens[::-1]

#     def save_combo_mapping(self):
#         """Saves the combo_to_id mapping to a JSON file."""
#         with open(self.combo_path, "w") as f:
#             json.dump(self.combo_to_id, f, indent=2)
#         print(f"✅ Combo mapping saved to {self.combo_path}")

#     def map_track_id_to_tokens(self, track_identifier: Union[int, str]) -> List[int]:
#         """
#         Maps a flexible track identifier to a list of four tokens.
#         Handles integers, zero-padded strings, full track IDs, and filenames.
#         """
#         match = re.match(r'^\d+', str(track_identifier))
#         if str(track_identifier).startswith('track_'):
#             track_id = str(track_identifier)
#         elif match:
#             track_num = int(match.group(0))
#             track_id = f"track_{track_num:07d}"
#         else:
#             raise ValueError(f"Could not parse a valid track number from '{track_identifier}'")

#         if track_id not in self._track_tags:
#             raise KeyError(f"Track ID not found in the dataset: '{track_id}'")

#         tags = self._track_tags[track_id]
#         key = self._tags_to_key(tags)
#         return self._key_to_tokens(key)

#     def map_tags_to_tokens(self, category_tags: Dict[str, List[str]]) -> List[int]:
#         """
#         Maps a dictionary of tags to a list of four tokens.
#         """
#         genres = sorted(set(category_tags.get("genre", [])))[:3]
#         instruments = sorted(set(category_tags.get("instrument", [])))[:3]
#         moods = sorted(set(category_tags.get("mood/theme", [])))[:3]

#         tags_list = [f"genre---{g}" for g in genres] + \
#                     [f"mood/theme---{m}" for m in moods] + \
#                     [f"instrument---{i}" for i in instruments]

#         key = self._tags_to_key(tags_list)
#         return self._key_to_tokens(key)

# def main():
#     """
#     Main function to execute the mapping process.
#     """
#     if not os.path.exists(TSV_PATH):
#         print(f"❌ Error: The file '{TSV_PATH}' was not found.")
#         return

#     print(f"📖 Reading content from '{TSV_PATH}'...")
#     with open(TSV_PATH, "r", encoding="utf-8") as f:
#         tsv_content = f.read()
#     print("✅ File read successfully.")

#     print("🧠 Initializing the mapper and creating the tag-to-token mapping...")
#     mapper = TrackTagMapper(tsv_content=tsv_content, combo_path=COMBO_PATH)
#     print("✅ Mapper initialized.")
#     mapper.save_combo_mapping()

#     print("\n--- 🚀 Testing the mapper with 10 random tracks to verify the fix ---")
#     track_ids = list(mapper._track_tags.keys())
#     # Add your specific failing track_id to the test samples to ensure it's fixed
#     test_sample_ids = random.sample(track_ids, 60) + ["track_1052139"]

#     for track_id in test_sample_ids:
#         try:
#             tokens_from_id = mapper.map_track_id_to_tokens(track_id)
#             original_tags = mapper._track_tags[track_id]

#             # Re-create the tag dictionary from the full list of tags
#             tags_dict = {"genre": [], "instrument": [], "mood/theme": []}
#             for tag in original_tags:
#                 if '---' in tag:
#                     category, value = tag.split("---", 1)
#                     if category in tags_dict:
#                         tags_dict[category].append(value)

#             tokens_from_tags = mapper.map_tags_to_tokens(tags_dict)
#             generated_key = mapper._tags_to_key(original_tags)

#             print(f"\nTrack: {track_id}")
#             print(f"  Original tags: {original_tags}")
#             print(f"  Generated key: '{generated_key}'")
#             print(f"  Tokens from track ID -> {tokens_from_id}")
#             print(f"  Tokens from tags ----> {tokens_from_tags}")

#             if tokens_from_id == tokens_from_tags:
#                 print("  ✅ Tokens match!")
#             else:
#                 print("  ❌ TOKEN MISMATCH!")
#                 print(f"  Key from tags dict: '{mapper._tags_to_key_from_dict(tags_dict)}'")

#         except (KeyError, ValueError) as e:
#             print(f"\n❌ Error processing track {track_id}: {e}")

# if __name__ == "__main__":
#     main()

# import csv
# import json
# import os
# import io
# from itertools import combinations, chain, product
# from tqdm import tqdm
# from typing import List, Dict, Set, Iterable, Iterator
# from math import comb

# # --- Configuration ---
# # Define the source TSV file and the destination for the new exhaustive mapping
# TSV_PATH = "/home/tzlillev/LLadaSMDM/mtg-jamendo-dataset/data/autotagging.tsv"
# EXHAUSTIVE_JSON_PATH = "/home/tzlillev/LLadaSMDM/mtg-jamendo-dataset/exhaustive_tag_mapping.json"

# def get_unique_tags_from_tsv_stream(tsv_path: str) -> Dict[str, List[str]]:
#     """
#     Parses the TSV file line-by-line to find all unique tags for each category,
#     avoiding loading the whole file into memory.

#     Args:
#         tsv_path: The path to the autotagging.tsv file.

#     Returns:
#         A dictionary where keys are categories and values are sorted lists of unique tags.
#     """
#     unique_tags: Dict[str, Set[str]] = {
#         "genre": set(),
#         "instrument": set(),
#         "mood/theme": set()
#     }

#     with open(tsv_path, "r", encoding="utf-8") as f:
#         reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
#         next(reader)  # Skip header

#         for row in tqdm(reader, desc="Parsing TSV to find unique tags"):
#             if not row:
#                 continue
#             # Tags are in the 6th column (index 5) onwards
#             tags = row[5:]
#             for tag in tags:
#                 if '---' in tag:
#                     try:
#                         category, value = tag.split("---", 1)
#                         if category in unique_tags:
#                             unique_tags[category].add(value)
#                     except ValueError:
#                         continue
    
#     # Return a dictionary with sorted lists for consistent ordering
#     return {category: sorted(list(tags)) for category, tags in unique_tags.items()}

# def generate_combinations_iterator(items: List[str], max_items: int) -> Iterator[tuple]:
#     """
#     Yields all possible combinations of items, from 0 up to max_items,
#     as an iterator to save memory.

#     Args:
#         items: A list of items to combine.
#         max_items: The maximum number of items in a combination.

#     Yields:
#         A tuple for each unique combination.
#     """
#     return chain.from_iterable(combinations(items, i) for i in range(max_items + 1))

# def count_combinations(item_count: int, max_items: int) -> int:
#     """Calculates the total number of combinations without generating them."""
#     return sum(comb(item_count, i) for i in range(max_items + 1))

# def create_exhaustive_mapping_optimized():
#     """
#     Main function to generate and save the tag mapping using a memory-efficient
#     streaming approach.
#     """
#     # --- Step 1: Extract unique tags by streaming the TSV file ---
#     if not os.path.exists(TSV_PATH):
#         print(f"❌ Error: The file '{TSV_PATH}' was not found.")
#         return

#     print("📖 Streaming TSV file to extract unique tags...")
#     unique_tags = get_unique_tags_from_tsv_stream(TSV_PATH)
#     print("✅ Unique tags extracted successfully.")
#     print(f"  - {len(unique_tags['genre'])} genres")
#     print(f"  - {len(unique_tags['instrument'])} instruments")
#     print(f"  - {len(unique_tags['mood/theme'])} moods/themes")

#     # --- Step 2: Create iterators for combinations ---
#     print("\n⚙️ Preparing combination generators (memory-efficient)...")
#     genre_combos_iter = generate_combinations_iterator(unique_tags['genre'], 2)
#     instrument_combos_iter = generate_combinations_iterator(unique_tags['instrument'], 3)
#     mood_combos_iter = generate_combinations_iterator(unique_tags['mood/theme'], 1)
    
#     # Calculate total combinations for the progress bar without storing them
#     num_genre_combos = count_combinations(len(unique_tags['genre']), 2)
#     num_instrument_combos = count_combinations(len(unique_tags['instrument']), 3)
#     num_mood_combos = count_combinations(len(unique_tags['mood/theme']), 1)
#     total_possible = num_genre_combos * num_instrument_combos * num_mood_combos
    
#     print(f"  - Found {num_genre_combos:,} possible genre combinations.")
#     print(f"  - Found {num_instrument_combos:,} possible instrument combinations.")
#     print(f"  - Found {num_mood_combos:,} possible mood/theme combinations.")
#     print(f"  - Total possible combinations to map: {total_possible:,}")
    
#     # --- Step 3: Stream-generate and write the final mapping to JSON ---
#     print(f"\n🔄💾 Stream-writing the final mapping to '{EXHAUSTIVE_JSON_PATH}'...")
#     next_id = 0
    
#     # Use itertools.product to combine the iterators efficiently
#     combo_product = product(genre_combos_iter, instrument_combos_iter, mood_combos_iter)

#     with open(EXHAUSTIVE_JSON_PATH, "w", encoding="utf-8") as f:
#         f.write("{\n")  # Start of JSON object
        
#         is_first_entry = True
#         progress_bar = tqdm(total=total_possible, desc="Generating and writing")

#         for g_combo, i_combo, m_combo in combo_product:
#             full_combo = sorted(list(g_combo) + list(i_combo) + list(m_combo))
            
#             # Skip the empty combination
#             if not full_combo:
#                 progress_bar.update(1)
#                 continue

#             key = "|||".join(full_combo)
            
#             # Write comma before adding a new entry (except for the first one)
#             if not is_first_entry:
#                 f.write(",\n")
            
#             # Use json.dumps to correctly handle escaping for the key
#             f.write(f'  {json.dumps(key)}: {next_id}')
            
#             is_first_entry = False
#             next_id += 1
#             progress_bar.update(1)
        
#         f.write("\n}")  # End of JSON object
    
#     progress_bar.close()
#     print("\n✅ Mapping created and saved successfully.")
#     print(f"🎉 All done! Your new mapping with {next_id:,} combinations is ready.")


# if __name__ == "__main__":
#     create_exhaustive_mapping_optimized()