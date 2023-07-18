# import os
# import openai
# import dotenv
import whisper

# # import API key from .env file
# dotenv.load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Use locally installed OpenAI whisper to transcribe audio file

def load_model(model_name: str = "base"):
    """
    Load a model of Whisper
    """
    return whisper.load_model(model_name)


def transcribe(audio_file):
    """
    Transcribe an audio file using OpenAI's Whisper.
    """
    # transcript = openai.Audio.transcribe("whisper-1", audio_file)

    model = load_model("base")
    transcript = model.transcribe(audio_file)

    return transcript

