import datetime
import os
import re
import sys

import pyperclip
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from whisper_API import transcribe


def save_audio_file(audio_bytes, file_extension):
    """
    Save audio bytes to a file with the specified extension.

    :param audio_bytes: Audio data in bytes
    :param file_extension: The extension of the output audio file
    :return: The name of the saved audio file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name


def transcribe_audio(file_path):
    """
    Transcribe the audio file at the specified path.

    :param file_path: The path of the audio file to transcribe
    :return: The transcribed text
    """
    # with open(file_path, "rb") as audio_file:
    #     transcript = transcribe(audio_file)
    transcript = transcribe(file_path)

    return transcript["text"]

def transcribe_button(transcribe_key: str, copykey: str):
    # Transcribe button action
    if st.button("Transcribe", key=transcribe_key):
        # Find the newest audio file
        audio_file_path = max(
            [f for f in os.listdir(".") if f.startswith("audio")],
            key=os.path.getctime,
        )

        # Transcribe the audio file
        transcript_text = transcribe_audio(audio_file_path)

        # Display the transcript
        st.header("Transcript")
        st.session_state["transcript"] = transcript_text
        st.write(transcript_text)
        # st.code(transcript_text)

        if st.button("Copy to Clipboard", key=copykey):
            pyperclip.copy(transcript_text)
            pyperclip.paste()
        
        # Save the transcript to a text file
        with open("transcript.txt", "w") as f:
            f.write(transcript_text)

        # Provide a download button for the transcript
        st.download_button("Download Transcript", transcript_text)

def delete_temp_audio_files():
    """
    Delete all temporary audio files.
    """
    # File name format: audio_20230718_234058.*
    # Regex: audio_\d{8}_\d{6}.*

    # Find all audio files using regex
    audio_files = [f for f in os.listdir(".") if re.match(r"audio_\d{8}_\d{6}.*", f)]

    # Delete all audio files
    for audio_file in audio_files:
        os.remove(audio_file)
        

def main():
    """
    Main function to run the Whisper Transcription app.
    """

    if "transcript" not in st.session_state:
        st.session_state["transcript"] = ""
    if "api_use" not in st.session_state:
        st.session_state["api_use"] = False

    st.title("Whisper Transcription")

    tab1, tab2, tab3 = st.tabs(["Record Audio", "Upload Audio", "Settings"])

    # Record Audio tab
    with tab1:
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            save_audio_file(audio_bytes, "mp3")

        transcribe_button("mic", "copy1")

    # Upload Audio tab
    with tab2:
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "wav", "m4a"])
        if audio_file:
            file_extension = audio_file.type.split('/')[1]
            save_audio_file(audio_file.read(), file_extension)
        
        transcribe_button("upload", "copy2")

    # Settings tab
    with tab3:
        st.header("Settings")
        
        if st.button("Toggle Whisper API use"):
            st.session_state["api_use"] = not st.session_state["api_use"]
            st.success(f"Whisper API use set to **{str(st.session_state['api_use']).upper()}**")

        # Delete temporary audio files
        if st.button("Delete Temporary Audio Files"):
            delete_temp_audio_files()
            st.success("Temporary audio files deleted.")
        

    st.write("---")

    if "transcript" in st.session_state:
        with st.expander("Recent Transcript", expanded=True):
            st.write(st.session_state["transcript"])

            if st.button("Copy to Clipboard", key="copy2"):
                    pyperclip.copy(st.session_state["transcript"])
                    pyperclip.paste()


if __name__ == "__main__":
    # Set up the working directory
    working_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(working_dir)

    # Run the main function
    main()
