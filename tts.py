"""
tts.py

Text-to-speech helpers used by the Streamlit app (app.py) to read
answers out loud.

- clean_text_for_voice : strips Markdown so speech sounds natural
- speak                : simple blocking pyttsx3 speech (legacy, single-shot)
- speak_async          : non-blocking pyttsx3 speech, cancels any speech in progress
- speak_edge_tts       : higher quality Edge TTS voice, played back as
                         an audio clip inside the Streamlit UI
"""

import os
import re
import asyncio
import tempfile
import threading

import pyttsx3
import edge_tts
import streamlit as st


# ============================================================
# Helper — Clean Markdown For Voice
# ============================================================

def clean_text_for_voice(text):
    """
    Convert an LLM Markdown response into clean natural
    language text for Text-To-Speech.

    UI:
        Original Markdown is preserved.

    Voice:
        Markdown symbols such as ##, **, links and bullets
        are removed.
    """

    if not text:

        return ""

    text = str(text)

    # --------------------------------------------------------
    # Remove code blocks completely
    # --------------------------------------------------------

    text = re.sub(
        r"```[\s\S]*?```",
        "",
        text
    )

    # --------------------------------------------------------
    # Remove Markdown headings
    # Example:
    # ## Answer
    # ### Summary
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE
    )

    # --------------------------------------------------------
    # Markdown links
    #
    # [Google](https://google.com)
    #
    # becomes:
    #
    # Google
    # --------------------------------------------------------

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # Bold
    # **text**
    # --------------------------------------------------------

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
        flags=re.DOTALL
    )

    # --------------------------------------------------------
    # Underline-style Markdown
    # __text__
    # --------------------------------------------------------

    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text,
        flags=re.DOTALL
    )

    # --------------------------------------------------------
    # Italic
    # *text*
    # --------------------------------------------------------

    text = re.sub(
        r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # Underscore italic
    # _text_
    # --------------------------------------------------------

    text = re.sub(
        r"(?<!_)_(?!\s)(.*?)(?<!\s)_",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # Inline code
    # `text`
    # --------------------------------------------------------

    text = re.sub(
        r"`([^`]*)`",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # Bullet points
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*[-*+]\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # --------------------------------------------------------
    # Numbered lists
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*\d+\.\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # --------------------------------------------------------
    # Remove remaining Markdown characters
    # --------------------------------------------------------

    text = text.replace("#", "")

    text = text.replace("**", "")

    text = text.replace("__", "")

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Voice Assistant Setup (pyttsx3 engines)
# ============================================================

_speech_lock = threading.Lock()

_current_engine = {
    "engine": None
}


def speak(text):
    """
    Simple blocking speech.
    Kept for the old single-shot flow.
    """

    local_engine = pyttsx3.init()

    local_engine.setProperty(
        "rate",
        170
    )

    local_engine.say(text)

    local_engine.runAndWait()

    local_engine.stop()


def speak_async(text):

    with _speech_lock:

        old_engine = (
            _current_engine["engine"]
        )

        if old_engine is not None:

            try:

                old_engine.stop()

            except Exception:

                pass


        new_engine = pyttsx3.init()

        new_engine.setProperty(
            "rate",
            170
        )

        _current_engine["engine"] = (
            new_engine
        )


    def run():

        try:

            new_engine.say(text)

            new_engine.runAndWait()

        except Exception:

            pass


    threading.Thread(
        target=run,
        daemon=True
    ).start()


async def _edge_tts_save(
    text,
    path,
    voice="en-US-AriaNeural"
):

    communicate = edge_tts.Communicate(
        text,
        voice=voice
    )

    await communicate.save(
        path
    )


def speak_edge_tts(
    text,
    container=None
):

    if not text:

        return

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    ) as f:

        mp3_path = f.name


    try:

        asyncio.run(
            _edge_tts_save(
                text,
                mp3_path
            )
        )

        target = (
            container
            if container is not None
            else st
        )

        with open(
            mp3_path,
            "rb"
        ) as audio_file:

            target.audio(
                audio_file.read(),
                format="audio/mp3",
                autoplay=True
            )

    finally:

        try:

            os.remove(
                mp3_path
            )

        except Exception:

            pass
