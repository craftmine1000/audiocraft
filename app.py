# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import warnings

import torch
import torchaudio
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen

MODEL = None  # Last used model
IS_BATCHED = "facebook/MusicGen" in os.environ.get('SPACE_ID', '')
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomitting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = MusicGen.get_pretrained(version)


def _do_predictions(texts, melodies, audios, re_prompt, duration, extra_prompt, prompt_period, gen_period, method, random_seed, seed, n_samples, progress=False, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, re_prompt_rate=re_prompt, interleaved_extra_prompt=extra_prompt, interleaved_prompt_period=prompt_period, interleaved_gen_period=gen_period, **gen_kwargs)
    print("new batch", len(texts), texts, [None if m is None else m for m in melodies])
    be = time.time()
    processed_melodies = []
    processed_audios = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            melody, sr = torchaudio.load(melody)
            melody = melody.to(MODEL.device)
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            print(melody.shape, torch.max(torch.abs(melody)))
            processed_melodies.append(melody)

    for audio in audios:
        if audio is None:
            processed_audios.append(None)
        else:
            audio, sr = torchaudio.load(audio)
            audio = audio.to(MODEL.device)
            if audio.dim() == 1:
                audio = audio[None]
            audio = audio[..., :int(sr * duration)]
            audio = convert_audio(audio, sr, target_sr, target_ac)
            print(audio.shape, torch.max(torch.abs(audio)))
            processed_audios.append(audio)

    if not random_seed:
        torch.manual_seed(seed)

    if method == 'generate_with_chroma':
        if any(m is not None for m in processed_melodies):
            outputs = getattr(MODEL, method)(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
            )
    elif method == 'generate':
        outputs = getattr(MODEL, method)(texts, progress=progress)
    elif method == 'generate_unconditional':
        outputs = getattr(MODEL, method)(int(n_samples), progress=progress)
    elif method == 'generate_continuation':
        outputs = getattr(MODEL, method)(torch.stack(processed_audios, dim=0), target_sr, texts, progress=progress)
    elif method == 'generate_continuation_with_chroma':
        outputs = getattr(MODEL, method)(torch.stack(processed_audios, dim=0), target_sr, processed_melodies, target_sr, texts, progress=progress)
    elif method == 'generate_continuation_continuous':
        outputs = getattr(MODEL, method)(torch.stack(processed_audios, dim=0), target_sr, texts, progress=progress)
    elif method == 'generate_continuation_with_chroma_continuous':
        outputs = getattr(MODEL, method)(torch.stack(processed_audios, dim=0), target_sr, processed_melodies, target_sr, texts, progress=progress)
    elif method == 'generate_continuation_interleaved':
        outputs = getattr(MODEL, method)(torch.stack(processed_audios, dim=0), target_sr, texts, progress=progress)


    outputs = outputs.detach().cpu().float()
    out_files = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            out_files.append(pool.submit(make_waveform, file.name))
    res = [out_file.result() for out_file in out_files]
    print("batch finished", len(texts), time.time() - be)
    return res


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('melody')
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return [res]


def predict_full(model, text, melody, audio, re_prompt, method, random_seed, seed, n_samples, duration, extra_prompt, prompt_period, gen_period, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    load_model(model)

    def _progress(generated, to_generate):
        progress((generated, to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    outs = _do_predictions(
        [text], [melody], [audio], re_prompt, duration, extra_prompt, prompt_period, gen_period, method, random_seed, seed, n_samples, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
    return outs[0]

def toggle_audio_src(choice, psfix=""):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone" + (" " + psfix if len(psfix) else ""))
    else:
        return gr.update(source="upload", value=None, label=(psfix + " " if len(psfix) else "") + "File")

def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MusicGen
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Text Prompt", interactive=True)
                with gr.Row():
                    melody = gr.Audio(source="upload", type="filepath", label="Melody Prompt", interactive=True, elem_id="melody-input")
                    audio = gr.Audio(source="upload", type="filepath", label="Audio Prompt", interactive=True, elem_id="audio-input")
                with gr.Row():
                    mic_radio_melody = gr.Radio(["file", "mic"], value="file", label="Condition on a melody (optional) File or Mic")
                    mic_radio_audio = gr.Radio(["file", "mic"], value="file", label="Prompt on audio (optional) File or Mic")
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    gr.Markdown(
                        """
                        * generate - generate from text prompt
                        * generate_unconditional - generate from nothing
                        * generate_with_chroma - generate from text prompt with melody condition from the melody prompt
                        * generate_continuation - generate from text prompt by continuing the audio prompt
                        * generate_continuation_with_chroma - generate from text prompt with melody condition from the melody prompt by continuing the audio prompt
                        * generate_continuation_continuous - generate from text prompt by continuing the audio prompt continuously
                        * generate_continuation_with_chroma_continuous - generate from text prompt with melody conditioning by continuing the audio prompt continuously
                        * generate_continuation_interleaved - generate from text prompt by continuing the audio prompt, prompt and generation get interleaved
                        """
                    )
                with gr.Row():
                    method = gr.Radio(
                        [
                            "generate",
                            "generate_unconditional",
                            "generate_with_chroma",
                            "generate_continuation",
                            "generate_continuation_with_chroma",
                            "generate_continuation_continuous",
                            "generate_continuation_with_chroma_continuous",
                            "generate_continuation_interleaved",
                        ],
                        label="Generation Method",
                        value="generate",
                        interactive=True
                    )
                with gr.Row():
                    model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=600, value=10, label="Duration", interactive=True)
                with gr.Row():
                    extra_prompt = gr.Number(label="Extra Prompting", interactive=True)
                    prompt_period = gr.Number(label="Prompt Period", interactive=True)
                    gen_period = gr.Number(label="Generation Period", interactive=True)
                with gr.Row():
                    random_seed = gr.Checkbox(label="Random Seed", value=True, interactive=True)
                    seed = gr.Number(label="Seed", interactive=True)
                with gr.Row():
                    re_prompt = gr.Slider(minimum=0.02, maximum=20, value=10, label="RePrompt Interval (continuous modes)", interactive=True)
                with gr.Row():
                    n_samples = gr.Number(label="Number Of Samples (generate_unconditional only)", value=1, interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                output = gr.Video(label="Generated Music")

        submit.click(predict_full, inputs=[model, text, melody, audio, re_prompt, method, random_seed, seed, n_samples, duration, extra_prompt, prompt_period, gen_period, topk, topp, temperature, cfg_coef], outputs=[output])
        
        mic_radio_melody.change(toggle_audio_src, [mic_radio_melody, gr.Text(value="Melody", visible=False, interactive=False)], [melody], queue=False, show_progress=False)
        mic_radio_audio.change(toggle_audio_src, [mic_radio_audio, gr.Text(value="Audio", visible=False, interactive=False)], [audio], queue=False, show_progress=False)

        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "",
                    None,
                    "generate_unconditional",
                    "medium",
                ],
                [
                    "A bach movement turning into an 80s pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "generate_continuation",
                    "melody",
                ],
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "generate_with_chroma",
                    "melody",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                    "generate_with_chroma",
                    "melody",
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                    "generate",
                    "medium",
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                    "generate_with_chroma",
                    "melody",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                    "generate",
                    "medium",
                ],
            ],
            inputs=[text, melody, method, model],
            outputs=[output]
        )
        gr.Markdown(
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            The model can generate up to 30 seconds of audio in one pass. It is now possible
            to extend the generation by feeding back the end of the previous chunk of audio.
            This can take a long time, and the model might lose consistency. The model might also
            decide at arbitrary positions that the song ends.

            **WARNING:** Choosing long durations will take a long time to generate (2min might take ~10min). An overlap of 12 seconds
            is kept with the previously generated chunk, and 18 "new" seconds are generated each time.

            We present 4 model variations:
            1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
            2. Small -- a 300M transformer decoder conditioned on text only.
            3. Medium -- a 1.5B transformer decoder conditioned on text only.
            4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.)

            When using `melody`, ou can optionaly provide a reference audio from
            which a broad melody will be extracted. The model will then try to follow both the description and melody provided.

            You can also use your own GPU or a Google Colab by following the instructions on our repo.
            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
            for more details.
            """
        )

        interface.queue().launch(**launch_kwargs)


def ui_batched(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # MusicGen

            This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Describe your music", lines=2, interactive=True)
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file", label="Condition on a melody (optional) File or Mic")
                        melody = gr.Audio(source="upload", type="numpy", label="File", interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Generate")
            with gr.Column():
                output = gr.Video(label="Generated Music")
        submit.click(predict_batched, inputs=[text, melody], outputs=[output], batch=True, max_batch_size=MAX_BATCH_SIZE)
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)
        gr.Examples(
            fn=predict_batched,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
                    "./assets/bach.mp3",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                ],
            ],
            inputs=[text, melody],
            outputs=[output]
        )
        gr.Markdown("""
        ### More details

        The model will generate 12 seconds of audio based on the description you provided.
        You can optionaly provide a reference audio from which a broad melody will be extracted.
        The model will then try to follow both the description and melody provided.
        All samples are generated with the `melody` model.

        You can also use your own GPU or a Google Colab by following the instructions on our repo.

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
        for more details.
        """)

        demo.queue(max_size=8 * 4).launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    # Show the interface
    if IS_BATCHED:
        ui_batched(launch_kwargs)
    else:
        ui_full(launch_kwargs)
