"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""
import os
import csv
import functools
import logging
import math
import json
import torch.nn.functional as F

import fsspec
import lightning
import torch
from timm.scheduler import CosineLRScheduler
from audiocraft.models.encodec import CompressionModel
import torchaudio
import csv
import random
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
  return sum(p.numel()
             for p in model.parameters()
             if p.requires_grad)

def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)

def update_and_save_csv(save_dict, csv_path):
  num_samples = len(save_dict['gen_ppl'])
  with fsspec.open(csv_path, 'a') as f:
    writer = csv.DictWriter(f, fieldnames=save_dict.keys())
    if fsspec_exists(csv_path) is False:
        writer.writeheader()
    for i in range(num_samples):
      row = {k: v[i] for k, v in save_dict.items()}
      writer.writerow(row)

class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


class Sampler:
  def __init__(self, shape):
    self.shape = shape

  def _sampling_noise(self):
    pass
  
  def _hard_sample(self, logits):
    pass

  def _soft_sample(self, logits):
    return 0

  def _process_logits(self, logits):
    return logits

  def sample(self, logits):
    logits = self._process_logits(logits)
    noise = self._sampling_noise()
    noise = noise[: logits.shape[0], :]
    logits = logits + noise.to(
      dtype=logits.dtype, device=logits.device)
    hard_sample = self._hard_sample(logits)
    soft_sample = self._soft_sample(logits)
    return soft_sample + (hard_sample - soft_sample).detach()


@functools.lru_cache(maxsize=None)
def log_n_choose_k(n, k):
  prior_loss = 0.0
  for i in range(k):
    prior_loss += (math.log(n - i) - math.log(k - i))
  return prior_loss


@functools.lru_cache(maxsize=None)
def log_n_permute_k(n, k):
  ans = 0.0
  for i in range(k):
    ans += math.log(n - i)
  return ans


class TopKSampler(Sampler):
  def __init__(self, k, shape, gamma_tau=1.0,
               noise_type='sog'):
    super().__init__(shape)
    self.k = k
    self.gamma_tau = gamma_tau
    self.num_betas = 10
    self.sampler = torch.distributions.gamma.Gamma(
      1 / k * torch.ones(self.num_betas, * self.shape), 1.0)
    self.noise_type = noise_type

  def _sampling_noise(self):
    if self.noise_type == 'sog':
      noise = self.sampler.sample()
      beta = self.k / torch.arange(1, self.num_betas + 1, 1,
                                  dtype=torch.float32)
      beta = beta[:, None, None, None]
      assert beta.ndim == noise.ndim
      s = noise / beta
      s = torch.sum(s, axis=0)
      s = s - math.log(self.num_betas)
      return self.gamma_tau * (s / self.k)
    elif self.noise_type == 'gumbel':
      return - (1e-10 - (torch.rand(* self.shape)
                         + 1e-10).log()).log()
    elif self.noise_type == 'deterministic':
      return torch.zeros(* self.shape)

  def _process_logits(self, logits):
    assert logits.ndim == 3
    return logits

  def _hard_sample(self, logits):
    thresholds, _ = torch.sort(logits, dim=-1)
    thresholds = thresholds[:, :, - self.k][:, :, None]
    return (logits >= thresholds).type(logits.dtype)

  def _soft_sample(self, logits):
    soft_top_k = logits - torch.mean(logits, dim=-1,
                                     keepdim=True)
    return soft_top_k / torch.norm(soft_top_k, dim=-1,
                                   keepdim=True)

class GaussianSampler:
  def __init__(self, constrain_logits):
    self.constrain_logits = constrain_logits

  def gaussian_params_from_logits(self, logits):
    assert logits.ndim == 3
    n = logits.shape[-1] // 2
    mu = logits[:, :, :n]
    log_var = logits[:, :, n:]
    if self.constrain_logits:
      # mu \in [0, 1]
      mu = torch.tanh(mu)
      # var \in [0, 1]
      # log_var \in (- \inf, 0]
      log_var = - torch.nn.functional.softplus(log_var)
    return mu, log_var

  def sample(self, x):
    mu, log_var = self.gaussian_params_from_logits(x)
    sigma = log_var.exp().sqrt()
    return mu + sigma * torch.randn_like(mu)
  

class EmptyTokenizer:
  def __init__(self, device="cuda"):
    # self.encodec_model = CompressionModel.get_pretrained('facebook/encodec_32khz').to(device)
    # self.encodec_model.eval()
    # self.encodec_model.normalize = False
    self.encodec_model = None
    # self.sample_rate = self.encodec_model.sample_rate
    self.vocab_size = 2053
    self.bos_token_id = 2052
    self.eos_token_id = 2051
    self.mask_token_id = 2049
    self.pad_token_id = 2050
    self.mask_index = 2049
    self.Output_test_wav_path = "/home/tzlillev/ProjectAudioMTG/bd3lms/test_wavs"
    os.makedirs(self.Output_test_wav_path, exist_ok=True)
    
  
  def batch_decode(self, x):
    if self.encodec_model is None:
        self.encodec_model = CompressionModel.get_pretrained('facebook/encodec_32khz').to(device)
        self.encodec_model.eval()
        self.encodec_model.normalize = False
        self.sample_rate = self.encodec_model.sample_rate
    print("Samples before decoding:", x[:, -300:-1])
    # Remove the first 12 tokens (guidance tokens + bos token)
    x = x[:, 12:]
    # if (x == self.pad_token_id).any():
    #     print("ERROR: Tokens contain padding tokens which should be handled before decoding")
        
    # # Check for and remove BOS/EOS tokens if present
    # if (x == self.bos_token_id).any():
    #     print("ERROR: Tokens contain BOS tokens which should be handled before decoding")
    # if (x == self.eos_token_id).any():
    #     print("ERROR: Tokens contain EOS tokens which should be handled before decoding")
   # print(f"Samples after filtering: {x[:, -300:-1]}")
    samples_before = x.clone()
    samples = x % self.encodec_model.cardinality
    assert torch.all(samples == samples_before % self.encodec_model.cardinality), "Modulo operation changed values unexpectedly"

    # Original samples
    with torch.autocast('cuda', dtype=torch.float32):
        audio_waveform = decode_type1(samples[0], self.encodec_model)
    wav_to_save = audio_waveform.squeeze(0).cpu()
    torchaudio.save(self.Output_test_wav_path + "/test_wav_original.wav", wav_to_save, self.sample_rate)

    # First shift
    shifted_samples = torch.roll(samples, shifts=-1, dims=-1)
    with torch.autocast('cuda', dtype=torch.float32):
        audio_waveform = decode_type1(shifted_samples[0], self.encodec_model)
    wav_to_save = audio_waveform.squeeze(0).cpu()
    torchaudio.save(self.Output_test_wav_path + "/test_wav_shift1.wav", wav_to_save, self.sample_rate)

    # Second shift  
    shifted_samples = torch.roll(samples, shifts=-2, dims=-1)
    with torch.autocast('cuda', dtype=torch.float32):
        audio_waveform = decode_type1(shifted_samples[0], self.encodec_model)
    wav_to_save = audio_waveform.squeeze(0).cpu()
    torchaudio.save(self.Output_test_wav_path + "/test_wav_shift2.wav", wav_to_save, self.sample_rate)

    # Third shift
    shifted_samples = torch.roll(samples, shifts=-3, dims=-1)
    with torch.autocast('cuda', dtype=torch.float32):
        audio_waveform = decode_type1(shifted_samples[0], self.encodec_model)
    wav_to_save = audio_waveform.squeeze(0).cpu()
    torchaudio.save(self.Output_test_wav_path + "/test_wav_shift3.wav", wav_to_save, self.sample_rate)

    return samples_before


def decode_type1(tokens, encodec_model):
    """
    Decodes tokens of shape (num_codebooks, sequence_length)
    """
    # The encodec model expects input of shape (batch_size, num_codebooks, seq_len)
    # The generated tokens are of shape (seq_len,).
    # We need to determine the number of codebooks from the model
    # and reshape the tokens accordingly.
    
    # Assuming batch size is 1.
    # The generated tokens are for a single audio stream.
    # The tokens are ordered [c1_t1, c2_t1, ..., cN_t1, c1_t2, ... ],
    # but the model expects [c1_t1, c1_t2, ...], [c2_t1, c2_t2, ...], ...
    # So we need to reshape.

    num_codebooks = encodec_model.num_codebooks
    
    # Ensure tokens is a tensor on the correct device
    device = next(encodec_model.parameters()).device
    tokens = torch.as_tensor(tokens, device=device, dtype=torch.long)
    empty_tokenizer = EmptyTokenizer()
    # Check for padding tokens
    #Remove the first 12 tokens (guidance tokens + bos token)
    tokens = tokens[12:]
    if (tokens == empty_tokenizer.pad_token_id).any():
        print("ERROR: Tokens contain padding tokens which should be handled before decoding")
        
    # Check for and remove BOS/EOS tokens if present s
    if (tokens == empty_tokenizer.bos_token_id).any():
        print("ERROR: Tokens contain BOS tokens which should be handled before decoding")
    if (tokens == empty_tokenizer.eos_token_id).any():
        print("ERROR: Tokens contain EOS tokens which should be handled before decoding")
    
    tokens = tokens[tokens != empty_tokenizer.bos_token_id]
    tokens = tokens[tokens != empty_tokenizer.eos_token_id]

    # Pad tokens to be divisible by num_codebooks
    remainder = tokens.shape[-1] % num_codebooks
    if remainder != 0:
        padding_needed = num_codebooks - remainder
        tokens = F.pad(tokens, (0, padding_needed))

    # Reshape from (seq_len,) to (1, num_codebooks, seq_len / num_codebooks)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0) # Add batch dimension

    # Reshape and transpose
    tokens = tokens.view(tokens.shape[0], -1, num_codebooks).transpose(1, 2)


    # Decode
    with torch.no_grad():
        reconstructed_waveform = encodec_model.decode(tokens)

    return reconstructed_waveform


class TagsTokenizer:
    """
    A standalone class to handle tokenization of music features.

    This class loads all necessary data upon initialization and provides a
    single method, get_tokens, to generate an 8-token list for a given
    track ID, artist ID, or feature dictionary.
    """
    def __init__(self, tsv_path='/home/tzlillev/LLadaSMDM/mtg-jamendo-dataset/data/autotagging.tsv', mappings_path='/home/tzlillev/ProjectAudioMTG/datasets_music/token_mappings.json'):
        """
        Initializes the tokenizer by loading all required data.
        
        Args:
            tsv_path (str): Path to the autotagging.tsv file.
            mappings_path (str): Path to the token_mappings.json file.
        """
        print("Initializing MusicTokenizer...")
        self.padding_token = 2050
        
        print("Loading token mappings...")
        self.mappings = self._load_token_mappings(mappings_path)
        
        print("Parsing track and artist data from TSV...")
        self.track_data, self.artist_data = self._parse_data(tsv_path)
        
        if self.mappings and self.track_data:
            print("\n✅ Tokenizer ready.\n" + "-"*25)
        else:
            print("\n❌ Tokenizer initialization failed. Please check file paths.")

    def _load_token_mappings(self, file_path):
        """Loads the pre-generated token mappings from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Mapping file not found at '{file_path}'.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{file_path}'.")
            return None

    def _parse_data(self, file_path):
        """
        Parses the TSV file to create dictionaries for tracks and artists.
        This version correctly handles rows with a variable number of tags.
        """
        tracks = {}
        artist_features_agg = defaultdict(lambda: {"mood": set(), "genre": set(), "instruments": set()})
        
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
                # Use csv.reader for robust handling of variable-length rows
                reader = csv.reader(tsvfile, delimiter='\t')
                # Skip the header row
                header = next(reader)
                
                for row in reader:
                    # Ensure the row has at least the minimum number of columns
                    if len(row) < 6:
                        continue

                    track_id = row[0]
                    artist_id = row[1]
                    
                    # All items from the 6th column onwards are considered tags
                    all_tags = row[5:]
                    
                    track_features = {"mood": [], "genre": [], "instruments": []}
                    for tag in all_tags:
                        tag = tag.strip()
                        if 'genre---' in tag:
                            feature = tag.split('---', 1)[1]
                            track_features["genre"].append(feature)
                            artist_features_agg[artist_id]["genre"].add(feature)
                        elif 'mood/theme---' in tag:
                            feature = tag.split('---', 1)[1]
                            track_features["mood"].append(feature)
                            artist_features_agg[artist_id]["mood"].add(feature)
                        elif 'instrument---' in tag:
                            feature = tag.split('---', 1)[1]
                            track_features["instruments"].append(feature)
                            artist_features_agg[artist_id]["instruments"].add(feature)
                    tracks[track_id] = track_features
        except FileNotFoundError:
            print(f"Error: Track data file not found at '{file_path}'.")
            return None, None
            
        artists = {
            artist_id: {
                "mood": sorted(list(feats["mood"])),
                "genre": sorted(list(feats["genre"])),
                "instruments": sorted(list(feats["instruments"])),
            }
            for artist_id, feats in artist_features_agg.items()
        }
        
        return tracks, artists

    def tokenize(self, input_data):
        """
        Generates an 8-token list based on input features, a track ID, or an artist ID.
        """
        if not self.mappings or not self.track_data:
            return {"error": "Tokenizer is not properly initialized."}

        features_to_tokenize = {}
        input_type = "Dictionary"

        if isinstance(input_data, int):
            input_type = "Track ID"
            track_id_str = f"track_{input_data:07d}"
            features_to_tokenize = self.track_data.get(track_id_str, {}).copy()
            if not features_to_tokenize:
                return {"error": f"Track ID '{track_id_str}' not found."}

        elif isinstance(input_data, str) and input_data.startswith('artist_'):
            input_type = "Artist ID"
            features_to_tokenize = self.artist_data.get(input_data, {}).copy()
            if not features_to_tokenize:
                return {"error": f"Artist ID '{input_data}' not found."}
                
        elif isinstance(input_data, dict):
            features_to_tokenize = input_data
        else:
            return {"error": "Invalid input. Provide an integer track ID, an artist ID string, or a dictionary."}

        if input_type in ["Track ID", "Artist ID"]:
            if len(features_to_tokenize.get("genre", [])) > 2:
                features_to_tokenize["genre"] = random.sample(features_to_tokenize["genre"], 2)
            if len(features_to_tokenize.get("mood", [])) > 2:
                features_to_tokenize["mood"] = random.sample(features_to_tokenize["mood"], 2)
            if len(features_to_tokenize.get("instruments", [])) > 4:
                features_to_tokenize["instruments"] = random.sample(features_to_tokenize["instruments"], 4)

        genre_map = self.mappings.get("genre_map", {})
        mood_map = self.mappings.get("mood_map", {})
        instrument_map = self.mappings.get("instrument_map", {})

        genre_tokens = [genre_map.get(g, self.padding_token) for g in features_to_tokenize.get("genre", [])]
        mood_tokens = [mood_map.get(m, self.padding_token) for m in features_to_tokenize.get("mood", [])]
        instrument_tokens = [instrument_map.get(i, self.padding_token) for i in features_to_tokenize.get("instruments", [])]

        padded_genres = (genre_tokens + [self.padding_token] * 2)[:2]
        padded_moods = (mood_tokens + [self.padding_token] * 2)[:2]
        padded_instruments = (instrument_tokens + [self.padding_token] * 4)[:4]

        final_tokens = padded_genres + padded_moods + padded_instruments
        return {
            "tokens": final_tokens,
            "features_used": features_to_tokenize
        }