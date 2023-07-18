import os

import dotenv
import openai
import streamlit as st
import whisper


def load_model(model_name: str = "base"):
    """
    Load a model of Whisper
    """
    return whisper.load_model(model_name)


def transcribe(audio_file, api_use: bool = False):
    """
    Transcribe an audio file using OpenAI's Whisper.
    """

    if api_use:
        # # import API key from .env file
        dotenv.load_dotenv()
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

        if OPENAI_API_KEY:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        else:
            st.error("API key not found. Please add it to the .env file.")
            
    else:
        model = load_model("base")
        transcript = model.transcribe(audio_file)

    return transcript

