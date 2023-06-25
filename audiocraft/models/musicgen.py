# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MusicGen. This will combine all the required components
and provide easy access to the generation API.
"""

import os
import typing as tp

import torch

from .encodec import CompressionModel
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model
from .loaders import load_compression_model, load_lm_model, HF_MODEL_CHECKPOINTS_MAP
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition
from ..utils.autocast import TorchAutocast


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]

ALWAYS_PROGRESS = True

class MusicGen:
    """MusicGen main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
    """
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: float = 30):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        self.max_duration = max_duration
        self.device = next(iter(lm.parameters())).device
        self.generation_params: dict = {}
        self.set_generation_params(duration=15)  # 15 seconds by default
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device.type, dtype=torch.float16)

    @property
    def frame_rate(self) -> int:
        """Roughly the number of AR steps per seconds."""
        return self.compression_model.frame_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio."""
        return self.compression_model.sample_rate

    @property
    def audio_channels(self) -> int:
        """Audio channels of the generated audio."""
        return self.compression_model.channels

    @staticmethod
    def get_pretrained(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            if not os.path.isfile(name) and not os.path.isdir(name):
                raise ValueError(
                    f"{name} is not a valid checkpoint name. "
                    f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
                )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              two_step_cfg: bool = False, extend_stride: float = 18,
                              re_prompt_rate: float = 1, batch_size: int = 4,
                              interleaved_prompt_period: float = 1, interleaved_gen_period: float = 1,
                              interleaved_extra_prompt: float = 1):
        """Set the generation parameters for MusicGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 30 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.re_prompt_rate = re_prompt_rate
        self.batch_size = batch_size
        self.interleaved_prompt_period = interleaved_prompt_period
        self.interleaved_gen_period = interleaved_gen_period
        self.interleaved_extra_prompt = interleaved_extra_prompt
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
        }

    def set_custom_progress_callback(self, progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None):
        """Override the default progress callback."""
        self._progress_callback = progress_callback

    def generate_unconditional(self, num_samples: int, progress: bool = False) -> torch.Tensor:
        """Generate samples in an unconditional manner.

        Args:
            num_samples (int): Number of samples to be generated.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        descriptions: tp.List[tp.Optional[str]] = [None] * num_samples
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        return self._generate_tokens(attributes, prompt_tokens, progress)

    def generate(self, descriptions: tp.List[str], progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        return self._generate_tokens(attributes, prompt_tokens, progress)

    def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        return self._generate_tokens(attributes, prompt_tokens, progress)

    def generate_continuation(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on audio prompts.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (tp.List[str], optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        assert prompt_tokens is not None
        return self._generate_tokens(attributes, prompt_tokens, progress)

    def generate_continuation_with_chroma(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              melody_wavs: MelodyType, melody_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False) -> torch.Tensor:
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)

        if descriptions is None:
            descriptions = [None] * len(prompt)

        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."
        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]

        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions,
                                                                        prompt=prompt,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is not None

        return self._generate_tokens(attributes, prompt_tokens, progress)

    def generate_continuation_continuous(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on audio prompts.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (tp.List[str], optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        old_batch = len(prompt)

        old_duration = self.duration
        self.duration = self.max_duration

        re_prompt_rate_sr = int(self.re_prompt_rate * self.sample_rate)
        re_prompt_mod_sr = int((self.duration - self.re_prompt_rate) * self.sample_rate)

        new_prompts = []
        for i in range(0, prompt.shape[-1], re_prompt_rate_sr):
            cut = prompt[:,:,max(0, i - re_prompt_mod_sr):i]
            missing = re_prompt_mod_sr - cut.shape[-1]
            cut = torch.nn.functional.pad(cut, (missing, 0))
            new_prompts.append(cut)
        prompt = torch.cat(new_prompts)
        print(prompt.shape)

        descriptions = descriptions * (len(prompt) // old_batch)

        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        assert prompt_tokens is not None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress, True)
        print(tokens.shape)

        #tokens = tokens.reshape(old_batch, -1)
        tokens = tokens.reshape(old_batch, 1, -1)
        print(tokens.shape)
        self.duration = old_duration
        return tokens

    def generate_continuation_with_chroma_continuous(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              melody_wavs: MelodyType, melody_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on audio prompts.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (tp.List[str], optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        old_batch = len(prompt)

        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."
        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]

        old_duration = self.duration
        self.duration = self.max_duration

        re_prompt_rate_sr = int(self.re_prompt_rate * self.sample_rate)
        re_prompt_mod_sr = int((self.duration - self.re_prompt_rate) * self.sample_rate)

        new_prompts = []
        for i in range(0, prompt.shape[-1], re_prompt_rate_sr):
            cut = prompt[:,:,max(0, i - re_prompt_mod_sr):i]
            missing = re_prompt_mod_sr - cut.shape[-1]
            cut = torch.nn.functional.pad(cut, (missing, 0))
            new_prompts.append(cut)
        prompt = torch.cat(new_prompts)

        new_melodies = []
        for melody in melody_wavs:
            tmp_mld = []
            for i in range(0, melody.shape[-1], re_prompt_rate_sr):
                cut = melody[:, i : i + re_prompt_rate_sr]
                #missing = re_prompt_mod_sr - cut.shape[-1]
                #cut = torch.nn.functional.pad(cut, (missing, 0))
                tmp_mld.append(cut)
            new_melodies.append(tmp_mld)

        interleaved_melodies = [mel for tup in zip(*new_melodies) for mel in tup]

        descriptions = descriptions * (len(prompt) // old_batch)

        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt, interleaved_melodies)
        assert prompt_tokens is not None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress, True)
        print(tokens.shape)

        #tokens = tokens.reshape(old_batch, -1)
        tokens = tokens.reshape(old_batch, 1, -1)
        print(tokens.shape)
        self.duration = old_duration
        return tokens

    def generate_continuation_interleaved(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on audio prompts. interleaving between prompt and generation

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (tp.List[str], optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        prompt_length = prompt.shape[2]

        old_duration = self.duration

        step_size = int((self.interleaved_prompt_period + self.interleaved_gen_period) * self.sample_rate)
        prime_samples = int(self.interleaved_extra_prompt * self.sample_rate)
        prompt_samples = int(self.interleaved_prompt_period * self.sample_rate)
        max_duration_samples = int(self.max_duration * self.sample_rate)

        new_prompt = prompt[:,:,:prime_samples]

        for i in range(prime_samples, prompt_length - step_size, step_size):
            new_prompt = torch.cat((new_prompt, prompt[:,:,i:i + prompt_samples]), dim=-1)

            temp_prompt = new_prompt[:,:,-max_duration_samples + prompt_samples:]
            self.duration = self.interleaved_gen_period + temp_prompt.shape[2] / self.sample_rate

            attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, temp_prompt)
            assert prompt_tokens is not None
            new_prompt = torch.cat((new_prompt, self._generate_tokens(attributes, prompt_tokens, progress, True)), dim=-1)

        self.duration = old_duration
        return new_prompt

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (tp.Optional[torch.Tensor], optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path='null_wav')  # type: ignore
        else:
            if self.name != "melody":
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        path='null_wav')  # type: ignore
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody.to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device))

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_chunks = []
            for i in range(0, len(prompt), self.batch_size):
                chunk, scale = self.compression_model.encode(prompt[i : i + self.batch_size])
                assert scale is None
                prompt_chunks.append(chunk)
            prompt_tokens = torch.cat(prompt_chunks)
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor],
                         progress: bool = False, remove_prompts: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
            prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress or ALWAYS_PROGRESS:
            callback = _progress_callback

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            token_chunks = []
            for i in range(0, len(attributes), self.batch_size):
                if prompt_tokens is not None:
                    p_chunk = prompt_tokens[i : i + self.batch_size]
                else:
                    p_chunk = None
                with self.autocast:
                    chunk = self.lm.generate(
                        p_chunk, attributes[i : i + self.batch_size],
                        callback=callback, max_gen_len=total_gen_len, remove_prompts=remove_prompts, **self.generation_params)
                token_chunks.append(chunk)
            gen_tokens = torch.cat(token_chunks)

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_wavs = [attr.wav['self_wav'] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                if not remove_prompts:
                    all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                for attr, ref_wav in zip(attributes, ref_wavs):
                    wav_length = ref_wav.length.item()
                    if wav_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(self.max_duration * self.sample_rate)
                    print(initial_position / self.sample_rate, wav_target_length / self.sample_rate)
                    positions = torch.arange(initial_position,
                                             initial_position + wav_target_length, device=self.device)
                    attr.wav['self_wav'] = WavCondition(
                        ref_wav[0][:, positions % wav_length],
                        torch.full_like(ref_wav[1], wav_target_length))
                token_chunks = []
                for i in range(0, len(attributes), self.batch_size):
                    if prompt_tokens is not None:
                        p_chunk = prompt_tokens[i : i + self.batch_size]
                    else:
                        p_chunk = None
                    with self.autocast:
                        chunk = self.lm.generate(
                            p_chunk, attributes[i : i + self.batch_size],
                            callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                    token_chunks.append(chunk)
                gen_tokens = torch.cat(token_chunks)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)

        # generate audio
        assert gen_tokens.dim() == 3
        audio_chunks = []
        for i in range(0, len(gen_tokens), self.batch_size):
            with torch.no_grad():
                chunk = self.compression_model.decode(gen_tokens[i : i + self.batch_size], None)
                audio_chunks.append(chunk)
        gen_audio = torch.cat(audio_chunks)
        return gen_audio
