import os

import dotenv
import openai
import streamlit as st

# import whisper
from faster_whisper import WhisperModel


def load_model(model_name: str = "base"):
    """
    Load a model of Whisper
    """
    # model = whisper.load_model(model_name)
    model = WhisperModel(model_name, device="cpu")
    return model


def transcribe(audio_file, api_use: bool = False):
    """
    Transcribe an audio file using OpenAI's Whisper.
    """

    if api_use:
        # import API key from .env file
        API_KEY = st.session_state["OPENAI_API_KEY"]

        if API_KEY is not None or API_KEY != "":
            try:
                openai.api_key = API_KEY
            except openai.error.AuthenticationError as err:
                st.error(
                    f"Please check settings, API key is **invalid**: {err}", icon="🔥"
                )
                st.stop()
            except openai.error.APIError as err:
                st.error(f"OpenAI API returned an API Error: {err}", icon="🔥")

            try:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
            except openai.error.APIConnectionError as err:
                st.error(f"Failed to connect to OpenAI API: {err}", icon="🔥")
        else:
            st.error("API key not found. Please input it first in the settings.")

    else:
        if "model" in st.session_state:
            model = load_model(st.session_state["model"])
        else:
            model = load_model("base")
        # transcript = model.transcribe(audio_file, fp16=False)
        segments, _ = model.transcribe(audio_file, word_timestamps=True)

        transcript = {}
        transcript["text"] = ""
        for segment in segments:
            for word in segment.words:
                transcript["text"] += word.word

    return transcript
