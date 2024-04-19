from __future__ import annotations

import os

import gradio as gr
import numpy as np
import torch
import librosa
from transformers import AutoProcessor, SeamlessM4TModel

from lang_list import (
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
    S2TT_TARGET_LANGUAGE_NAMES,
    T2TT_TARGET_LANGUAGE_NAMES,
    TEXT_SOURCE_LANGUAGE_NAMES,
    LANG_TO_SPKR_ID,
)

CACHE_EXAMPLES = os.getenv("CACHE_EXAMPLES") == "1"

TASK_NAMES = [
    "S2ST (Speech to Speech translation)",
    "S2TT (Speech to Text translation)",
    "T2ST (Text to Speech translation)",
    "T2TT (Text to Text translation)",
    "ASR (Automatic Speech Recognition)",
]
AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
DEFAULT_TARGET_LANGUAGE = "French"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



processor = AutoProcessor.from_pretrained("ylacombe/hf-seamless-m4t-large")
model = SeamlessM4TModel.from_pretrained("ylacombe/hf-seamless-m4t-large").to(device)


def predict(
    task_name: str,
    audio_source: str,
    input_audio_mic: str | None,
    input_audio_file: str | None,
    input_text: str | None,
    source_language: str | None,
    target_language: str,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    task_name = task_name.split()[0]
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

    if task_name in ["S2ST", "S2TT", "ASR"]:
        if audio_source == "microphone":
            input_data, org_sr = librosa.load(input_audio_mic, sr=None)
        else:
            input_data, org_sr = librosa.load(input_audio_file, sr=None)
        
        new_sr = int(AUDIO_SAMPLE_RATE)
        new_arr = librosa.resample(input_data, orig_sr=org_sr, target_sr=new_sr)
        max_length = int(MAX_INPUT_AUDIO_LENGTH * new_sr)
        if len(new_arr) > max_length:
            new_arr = new_arr[:max_length]
            gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")

        input_data = processor(audios=new_arr, sampling_rate=new_sr, return_tensors="pt").to(device)
    else:
        input_data = processor(text=input_text, src_lang=source_language_code, return_tensors="pt").to(device)


    if task_name in ["S2TT", "T2TT"]:
        tokens_ids = model.generate(**input_data, generate_speech=False, tgt_lang=target_language_code, num_beams=5, do_sample=True)[0].cpu().squeeze().detach().tolist()
    else:
        output = model.generate(**input_data, return_intermediate_token_ids=True, tgt_lang=target_language_code, num_beams=5, do_sample=True, spkr_id=LANG_TO_SPKR_ID[target_language_code][0])
        
        waveform = output.waveform.cpu().squeeze().detach().numpy()
        tokens_ids = output.sequences.cpu().squeeze().detach().tolist()

    text_out = processor.decode(tokens_ids, skip_special_tokens=True)

    if task_name in ["S2ST", "T2ST"]:
        return (AUDIO_SAMPLE_RATE, waveform), text_out
    else:
        return None, text_out


def process_s2st_example(input_audio_file: str, target_language: str) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="S2ST",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
    )


def process_s2tt_example(input_audio_file: str, target_language: str) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="S2TT",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
    )


def process_t2st_example(
    input_text: str, source_language: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="T2ST",
        audio_source="",
        input_audio_mic=None,
        input_audio_file=None,
        input_text=input_text,
        source_language=source_language,
        target_language=target_language,
    )


def process_t2tt_example(
    input_text: str, source_language: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="T2TT",
        audio_source="",
        input_audio_mic=None,
        input_audio_file=None,
        input_text=input_text,
        source_language=source_language,
        target_language=target_language,
    )


def process_asr_example(input_audio_file: str, target_language: str) -> tuple[tuple[int, np.ndarray] | None, str]:
    return predict(
        task_name="ASR",
        audio_source="file",
        input_audio_mic=None,
        input_audio_file=input_audio_file,
        input_text=None,
        source_language=None,
        target_language=target_language,
    )


def update_audio_ui(audio_source: str) -> tuple[dict, dict]:
    mic = audio_source == "microphone"
    return (
        gr.update(visible=mic, value=None),  # input_audio_mic
        gr.update(visible=not mic, value=None),  # input_audio_file
    )


def update_input_ui(task_name: str) -> tuple[dict, dict, dict, dict]:
    task_name = task_name.split()[0]
    if task_name == "S2ST":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True, choices=S2ST_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    elif task_name == "S2TT":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True, choices=S2TT_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    elif task_name == "T2ST":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True, choices=S2ST_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    elif task_name == "T2TT":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True, choices=T2TT_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    elif task_name == "ASR":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True, choices=S2TT_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


def update_output_ui(task_name: str) -> tuple[dict, dict]:
    task_name = task_name.split()[0]
    if task_name in ["S2ST", "T2ST"]:
        return (
            gr.update(visible=True, value=None),  # output_audio
            gr.update(value=None),  # output_text
        )
    elif task_name in ["S2TT", "T2TT", "ASR"]:
        return (
            gr.update(visible=False, value=None),  # output_audio
            gr.update(value=None),  # output_text
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


def update_example_ui(task_name: str) -> tuple[dict, dict, dict, dict, dict]:
    task_name = task_name.split()[0]
    return (
        gr.update(visible=task_name == "S2ST"),  # s2st_example_row
        gr.update(visible=task_name == "S2TT"),  # s2tt_example_row
        gr.update(visible=task_name == "T2ST"),  # t2st_example_row
        gr.update(visible=task_name == "T2TT"),  # t2tt_example_row
        gr.update(visible=task_name == "ASR"),  # asr_example_row
    )


with gr.Blocks(css="styles.css") as demo:
    with gr.Group():
        task_name = gr.Dropdown(
            label="Task",
            choices=TASK_NAMES,
            value=TASK_NAMES[0],
        )
        with gr.Row():
            source_language = gr.Dropdown(
                label="Source language",
                choices=TEXT_SOURCE_LANGUAGE_NAMES,
                value="English",
                visible=False,
            )
            target_language = gr.Dropdown(
                label="Target language",
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            )
        with gr.Row() as audio_box:
            audio_source = gr.Radio(
                label="Audio source",
                choices=["file", "microphone"],
                value="file",
            )
            input_audio_mic = gr.Audio(
                label="Input speech",
                type="filepath",
                sources="microphone",
                visible=False,
            )
            input_audio_file = gr.Audio(
                label="Input speech",
                type="filepath",
                sources="upload",
                visible=True,
            )
        input_text = gr.Textbox(label="Input text", visible=False)
        btn = gr.Button("Translate")
        with gr.Column():
            output_audio = gr.Audio(
                label="Translated speech",
                autoplay=False,
                streaming=False,
                type="numpy",
            )
            output_text = gr.Textbox(label="Translated text")

    with gr.Row(visible=True) as s2st_example_row:
        s2st_examples = gr.Examples(
            examples=[
                ["assets/sample_input.mp3", "French"],
                ["assets/sample_input.mp3", "Mandarin Chinese"],
                ["assets/sample_input_2.mp3", "Hindi"],
            ],
            inputs=[input_audio_file, target_language],
            outputs=[output_audio, output_text],
            fn=process_s2st_example,
            cache_examples=CACHE_EXAMPLES,
        )
    with gr.Row(visible=False) as s2tt_example_row:
        s2tt_examples = gr.Examples(
            examples=[
                ["assets/sample_input.mp3", "French"],
                ["assets/sample_input.mp3", "Mandarin Chinese"],
                ["assets/sample_input_2.mp3", "Hindi"],
            ],
            inputs=[input_audio_file, target_language],
            outputs=[output_audio, output_text],
            fn=process_s2tt_example,
            cache_examples=CACHE_EXAMPLES,
        )
    with gr.Row(visible=False) as t2st_example_row:
        t2st_examples = gr.Examples(
            examples=[
                ["My favorite animal is the elephant.", "English", "French"],
                ["My favorite animal is the elephant.", "English", "Mandarin Chinese"],
                [
                    "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                    "English",
                    "Hindi",
                ],
            ],
            inputs=[input_text, source_language, target_language],
            outputs=[output_audio, output_text],
            fn=process_t2st_example,
            cache_examples=CACHE_EXAMPLES,
        )
    with gr.Row(visible=False) as t2tt_example_row:
        t2tt_examples = gr.Examples(
            examples=[
                ["My favorite animal is the elephant.", "English", "French"],
                ["My favorite animal is the elephant.", "English", "Mandarin Chinese"],
                [
                    "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
                    "English",
                    "Hindi",
                ],
            ],
            inputs=[input_text, source_language, target_language],
            outputs=[output_audio, output_text],
            fn=process_t2tt_example,
            cache_examples=CACHE_EXAMPLES,
        )
    with gr.Row(visible=False) as asr_example_row:
        asr_examples = gr.Examples(
            examples=[
                ["assets/sample_input.mp3", "English"],
                ["assets/sample_input_2.mp3", "English"],
            ],
            inputs=[input_audio_file, target_language],
            outputs=[output_audio, output_text],
            fn=process_asr_example,
            cache_examples=CACHE_EXAMPLES,
        )

    audio_source.change(
        fn=update_audio_ui,
        inputs=audio_source,
        outputs=[
            input_audio_mic,
            input_audio_file,
        ],
        queue=False,
        api_name=False,
    )
    task_name.change(
        fn=update_input_ui,
        inputs=task_name,
        outputs=[
            audio_box,
            input_text,
            source_language,
            target_language,
        ],
        queue=False,
        api_name=False,
    ).then(
        fn=update_output_ui,
        inputs=task_name,
        outputs=[output_audio, output_text],
        queue=False,
        api_name=False,
    ).then(
        fn=update_example_ui,
        inputs=task_name,
        outputs=[
            s2st_example_row,
            s2tt_example_row,
            t2st_example_row,
            t2tt_example_row,
            asr_example_row,
        ],
        queue=False,
        api_name=False,
    )

    btn.click(
        fn=predict,
        inputs=[
            task_name,
            audio_source,
            input_audio_mic,
            input_audio_file,
            input_text,
            source_language,
            target_language,
        ],
        outputs=[output_audio, output_text],
        api_name="run",
    )

    html_button = gr.Button(
        value="Go to Home Page",
        link="http://127.0.0.1:5000",
    )
demo.queue(max_size=50).launch()

# Linking models to the space
# 'facebook/seamless-m4t-large'
# 'facebook/SONAR'