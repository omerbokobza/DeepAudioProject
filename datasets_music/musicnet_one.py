import torch
import os
import datasets
import torch.utils.data as data

class TokenizedMusicNetOne(data.Dataset):
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

        # Only load the specific file 1727.pt for overfitting
        target_file = "1727.pt"
        target_path = os.path.join(self.root, target_file)
        
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target file '{target_file}' not found in '{self.root}'.")

        self.files = [target_file]

        # Load the single token file and keep as plain Python list so that we
        # can build a HuggingFace `Dataset`.
        token_sequences = []
        path = os.path.join(self.root, target_file)
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
        token_sequences.append(tokens.tolist())

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