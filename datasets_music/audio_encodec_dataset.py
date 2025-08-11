import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.utils.data as data
import torchaudio

# Encodec imports (installed as local package in this repo)
from encodec import EncodecModel
from encodec.utils import convert_audio


class EncodecAudioDataset(data.Dataset):
    """Dataset that takes raw audio files (wav/mp3/etc.), encodes them with
    Meta's Encodec neural codec and serves the resulting discrete codebook
    indices as token sequences.

    The encoded tensor for one file is flattened to 1-D so it looks exactly
    like a text token stream to the BD3LM loader.
    After a file is encoded once the tokens are cached next to the original
    audio under the same name but with a ``.pt`` extension so subsequent
    epochs start instantly.
    """

    AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    def __init__(
        self,
        root: str,
        *,
        sample_rate: int = 24_000,
        bandwidth: float = 12.0,
        encodec_model: EncodecModel | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

        # Gather audio &/or cached token files in a deterministic order
        self.files: List[Path] = sorted(
            p for p in self.root.iterdir() if p.suffix.lower() in self.AUDIO_EXTENSIONS or p.suffix == ".pt"
        )
        if not self.files:
            raise RuntimeError(f"No audio or .pt files found under {self.root}")

        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self._model = encodec_model  # can be shared between workers if given

    # ------------------------------------------------------------------
    # torch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        if path.suffix == ".pt":
            return torch.load(path).long()

        # Encode audio on the fly → tokens
        tokens = self._encode_audio(path)
        # Cache for future use
        torch.save(tokens, path.with_suffix(".pt"))
        # Replace entry in self.files so next call skips encoding
        self.files[idx] = path.with_suffix(".pt")
        return tokens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_encodec(self) -> EncodecModel:
        if self._model is not None:
            return self._model
        if self.sample_rate == 24_000:
            model = EncodecModel.encodec_model_24khz()
        elif self.sample_rate == 48_000:
            model = EncodecModel.encodec_model_48khz()
        else:
            raise ValueError("Encodec models are available only for 24 kHz and 48 kHz.")
        model.set_target_bandwidth(self.bandwidth)
        model.eval()
        self._model = model
        return model

    def _encode_audio(self, wav_path: Path) -> torch.LongTensor:
        wav, sr = torchaudio.load(str(wav_path))
        # Convert to mono
        wav = wav.mean(0, keepdim=True)
        model = self._get_encodec()
        if sr != model.sample_rate:
            wav = convert_audio(wav, sr, model.sample_rate, 1)
        with torch.no_grad():
            frames = model.encode(wav.unsqueeze(0))  # returns list of (codes, scale)
        tokens = [codes.squeeze(0).view(-1) for codes, _ in frames]
        return torch.cat(tokens).long() 