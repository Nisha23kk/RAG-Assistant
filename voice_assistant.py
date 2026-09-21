"""
============================================================
VOICE ASSISTANT  (fixed build)
============================================================

What was changed and WHY:

1.  MICROPHONE (your actual bug)
    A noise floor of 0.00001 means the stream returns digital
    silence. The old code built a threshold on top of a dead
    device and then waited forever. Now the app probes every
    input device, picks the loudest working one, applies a
    software gain, prints a live level meter, and tells you
    plainly when the device itself is dead.

2.  CRASH-ON-IMPORT
    ChatGroq and rag_engine were constructed at module import.
    A missing GROQ_API_KEY or any error inside rag_engine made
    the whole file unimportable. Both are lazy now.

3.  THREAD-SAFE TTS
    pyttsx3 is not thread safe. Reminders fired from a thread
    could deadlock the engine. All speech now goes through a
    lock.

4.  SEARCH SAFETY NET
    _DDGS_ERRORS was narrowed to DDGS-only exceptions, so any
    other error escaped web_search and crashed the caller.
    Exception is back as the final catch-all.

5.  EXIT COMMANDS
    Whisper returns "exit." with punctuation, so the exit list
    never matched. Commands are now normalised first.

6.  PUSH TO TALK
    Set PUSH_TO_TALK = True to bypass voice detection entirely
    when you just need the thing to work.

------------------------------------------------------------
FIRST RUN:
    python voice_assistant.py --devices     (list microphones)
    python voice_assistant.py --mic         (test capture)
    python voice_assistant.py               (run assistant)
------------------------------------------------------------
"""

import os
import re
import sys
import subprocess
import webbrowser
import datetime
import threading
import time

import numpy as np
import sounddevice as sd

from rapidfuzz import fuzz
from dotenv import load_dotenv
from faster_whisper import WhisperModel


# ============================================================
# ENVIRONMENT
# ============================================================
# Loaded before anything reads a key. If your .env sits next
# to this file but you run from another folder, the explicit
# path below still finds it.
# ============================================================

_ENV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".env"
)

if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()


if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY is not set.")
    print("Voice + website commands will still work.")
    print("Document / web answering will not.")


# ============================================================
# WEB SEARCH BACKEND
# ============================================================
# `duckduckgo_search` was RENAMED to `ddgs`. The old package
# still installs but is heavily rate limited, which is why web
# search silently returned nothing.
#
#     pip install -U ddgs
#     pip uninstall duckduckgo_search
# ============================================================

DDGS = None

# Always keep Exception last. Narrowing this list was a bug:
# a changed ddgs signature raises TypeError, which is NOT a
# DDGS exception, and it escaped web_search entirely.
_DDGS_ERRORS = (Exception,)

try:

    from ddgs import DDGS  # new package

    try:

        from ddgs.exceptions import (
            DDGSException,
            RatelimitException,
            TimeoutException
        )

        _DDGS_ERRORS = (
            RatelimitException,
            TimeoutException,
            DDGSException,
            Exception
        )

    except Exception:
        pass

except ImportError:

    try:

        from duckduckgo_search import DDGS  # legacy fallback

        print(
            "WARNING: using deprecated 'duckduckgo_search'. "
            "Run: pip install -U ddgs"
        )

    except ImportError:

        print(
            "ERROR: no search package installed. "
            "Run: pip install -U ddgs"
        )


# ============================================================
# RAG ENGINE  (lazy)
# ============================================================
# Importing rag_engine at module load meant ANY error inside
# it killed this file before a single line ran. Now the import
# is deferred and its failure is reported, not fatal.
# ============================================================

class RateLimitError(Exception):
    """Local stand-in used when rag_engine cannot be imported."""


_rag_engine = None
_rag_import_error = None
_rag_tried = False


def get_rag_engine():

    global _rag_engine, _rag_import_error, _rag_tried

    if _rag_tried:
        return _rag_engine

    _rag_tried = True

    try:

        import rag_engine as _module

        _rag_engine = _module

    except Exception as e:

        _rag_import_error = e

        print("=" * 60)
        print("rag_engine could not be imported.")
        print("Reason:", e)
        print("System commands still work. Q&A does not.")
        print("=" * 60)

        _rag_engine = None

    return _rag_engine


def get_rate_limit_error():
    """Return rag_engine's RateLimitError if available."""

    rag = get_rag_engine()

    return getattr(rag, "RateLimitError", RateLimitError)


# ============================================================
# GROQ LLM  (lazy, and optional)
# ============================================================
# Kept only for parity with the old standalone assistant. The
# actual question-answering always goes through rag_engine, so
# this must never be allowed to crash startup.
# ============================================================

_llm = None
_llm_tried = False


def get_llm():

    global _llm, _llm_tried

    if _llm_tried:
        return _llm

    _llm_tried = True

    key = os.getenv("GROQ_API_KEY")

    if not key:
        print("Groq client skipped: no GROQ_API_KEY.")
        return None

    try:

        from langchain_groq import ChatGroq

        _llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=key
        )

    except Exception as e:

        print("Groq client init failed:", e)

        _llm = None

    return _llm


# ============================================================
# TEXT TO SPEECH  (thread safe)
# ============================================================
# pyttsx3 is NOT thread safe. A reminder firing mid-sentence
# used to raise "run loop already started" or freeze. The lock
# serialises every call.
# ============================================================

import pyttsx3

_tts_lock = threading.Lock()

_engine = None
_engine_tried = False


def get_tts_engine():

    global _engine, _engine_tried

    if _engine_tried:
        return _engine

    _engine_tried = True

    try:

        _engine = pyttsx3.init()

        _engine.setProperty("rate", 170)
        _engine.setProperty("volume", 1.0)

        voices = _engine.getProperty("voices")

        if voices:
            _engine.setProperty("voice", voices[0].id)

    except Exception as e:

        print("TTS init failed (text output only):", e)

        _engine = None

    return _engine


def speak(text):

    if not text:
        return

    text = str(text)

    print("\nAssistant:", text)

    engine = get_tts_engine()

    if engine is None:
        return

    with _tts_lock:

        try:

            engine.say(text)
            engine.runAndWait()

        except Exception as e:

            print("TTS Error:", e)


# ============================================================
# AUDIO / WHISPER CONFIG
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

WHISPER_MODEL = "small"

START_TIMEOUT = 6.0
MAX_RECORD_SECONDS = 15.0

SILENCE_DURATION = 1.2
CHUNK_DURATION = 0.1

# 0.004 was far above what a normal laptop mic produces once
# gain is applied. The real floor is measured at runtime now.
MIN_RMS_THRESHOLD = 0.0015

PRE_ROLL_DURATION = 0.6

# ----- TUNE THESE IF NEEDED --------------------------------
# None = auto-pick the loudest working device on first use.
# int  = force a device index (see: python voice_assistant.py --devices)
INPUT_DEVICE = None

# Software gain. Raise to 8.0 or 15.0 for a very quiet mic.
INPUT_GAIN = 4.0

# True = record a fixed window after you press ENTER.
# No voice detection at all, so it cannot fail to trigger.
PUSH_TO_TALK = False

PTT_SECONDS = 6.0
# -----------------------------------------------------------


# ============================================================
# DOMAIN PROMPT
# ============================================================
# The "open ..." phrases help Whisper recognise website names.
# ============================================================

INITIAL_PROMPT = (
    "software development, AI, machine learning, "
    "RAG, documents, PDF, certifications, "
    "LangChain, FAISS, Python, Streamlit, "
    "Groq, embeddings, "
    "open YouTube, open Google, open Gmail, "
    "open GitHub, open LinkedIn, open Naukri"
)


# ============================================================
# CONVERSATION CONTEXT
# ============================================================

conversation_context = {
    "last_command": "",
    "last_intent": "",
    "last_topic": "",
    "last_website": "",
    "last_question": ""
}


def update_context(command, intent="", topic="", website=""):

    conversation_context["last_command"] = command

    if intent:
        conversation_context["last_intent"] = intent

    if topic:
        conversation_context["last_topic"] = topic

    if website:
        conversation_context["last_website"] = website

    conversation_context["last_question"] = command


# ============================================================
# INPUT DEVICE SELECTION  (the core fix)
# ============================================================

_device_ready = False


def list_input_devices():

    devices = []

    try:
        all_devices = sd.query_devices()
    except Exception as e:
        print("Could not query audio devices:", e)
        return devices

    for index, info in enumerate(all_devices):

        if info.get("max_input_channels", 0) > 0:
            devices.append((index, info.get("name", "unknown")))

    return devices


def print_input_devices():

    devices = list_input_devices()

    print("=" * 70)
    print("INPUT DEVICES")
    print("=" * 70)

    if not devices:
        print("None found. Your mic is not visible to PortAudio.")
        return

    for index, name in devices:
        print(f"[{index}] {name}")

    print("=" * 70)
    print("Set INPUT_DEVICE = <index> near the top of this file.")
    print("=" * 70)


def probe_device(index, seconds=0.6):
    """Peak amplitude for a device. 0.0 means it returns silence."""

    try:

        audio = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=index
        )

        sd.wait()

        return float(np.max(np.abs(audio)))

    except Exception:

        return -1.0


def print_device_help():

    print("-" * 60)
    print("Every microphone returned silence.")
    print("This is an OS / driver issue, not a code issue:")
    print("  1. Windows Settings > Privacy > Microphone")
    print("     -> allow desktop apps to access the microphone")
    print("  2. Sound > Recording > set your mic as Default Device")
    print("  3. Mic Properties > Levels > raise to 80-100")
    print("  4. Close Zoom / Teams / Discord - they can lock the mic")
    print("-" * 60)


def ensure_input_device(force_reprobe=False):
    """
    Picks a working microphone once, then caches the choice.

    A dead device returns pure zeros, which produced exactly
    the 0.00001 noise floor you were seeing.
    """

    global INPUT_DEVICE, _device_ready

    if _device_ready and not force_reprobe:
        return INPUT_DEVICE

    _device_ready = True

    devices = list_input_devices()

    if not devices:

        print("ERROR: no input devices found at all.")
        print("Check that your mic is plugged in and enabled.")

        INPUT_DEVICE = None
        return None

    # A forced device is trusted, but verified to exist.
    if INPUT_DEVICE is not None:

        valid = [i for i, _ in devices]

        if INPUT_DEVICE in valid:

            try:
                sd.default.device = (INPUT_DEVICE, None)
            except Exception:
                pass

            print("Using forced input device:", INPUT_DEVICE)

            return INPUT_DEVICE

        print("Forced device not found. Falling back to auto-detect.")

        INPUT_DEVICE = None

    print("=" * 60)
    print("Probing microphones - please make some noise...")
    print("=" * 60)

    best_index = None
    best_peak = 0.0

    for index, name in devices:

        peak = probe_device(index)

        if peak < 0:
            status = "UNAVAILABLE"
        elif peak == 0.0:
            status = "DEAD (pure silence)"
        else:
            status = f"peak={peak:.6f}"

        print(f"[{index}] {name}  ->  {status}")

        if peak > best_peak:

            best_peak = peak
            best_index = index

    if best_index is None or best_peak <= 0.0:

        print_device_help()

        INPUT_DEVICE = None
        return None

    INPUT_DEVICE = best_index

    try:
        sd.default.device = (INPUT_DEVICE, None)
    except Exception:
        pass

    print("-" * 60)
    print(f"Selected device [{INPUT_DEVICE}] (peak {best_peak:.6f})")

    if best_peak < 0.01:
        print("This device is very quiet. Consider raising INPUT_GAIN.")

    print("-" * 60)

    return INPUT_DEVICE


# ============================================================
# AUDIO HELPERS
# ============================================================

def calculate_rms(audio):

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return 0.0

    return float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )


def apply_gain(audio, gain=None):

    if gain is None:
        gain = INPUT_GAIN

    audio = np.asarray(audio, dtype=np.float32)

    if gain == 1.0:
        return audio

    return np.clip(audio * float(gain), -1.0, 1.0)


def normalize_audio(audio):

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return audio

    # Remove DC offset
    audio = audio - np.mean(audio)

    peak = float(np.max(np.abs(audio)))

    # The old version only scaled DOWN loud audio. Quiet speech
    # stayed quiet and Whisper heard nothing. Now it scales both
    # directions.
    if peak > 0.0001:
        audio = (audio / peak) * 0.95

    return audio.astype(np.float32)


# ============================================================
# NOISE CALIBRATION
# ============================================================

def calculate_noise_floor(duration=0.8):

    try:

        print("Calibrating microphone... please stay silent.")

        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=INPUT_DEVICE
        )

        sd.wait()

        return calculate_rms(
            apply_gain(audio.flatten())
        )

    except Exception as e:

        print("Noise calibration error:", e)

        return 0.005


# ============================================================
# PUSH TO TALK  (no voice detection, cannot fail to trigger)
# ============================================================

def record_push_to_talk(seconds=None):

    if seconds is None:
        seconds = PTT_SECONDS

    ensure_input_device()

    try:

        input(f"\nPress ENTER, then speak for {seconds:.0f} seconds...")

        print("Recording NOW.")

        audio = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=INPUT_DEVICE
        )

        sd.wait()

        audio = apply_gain(audio.flatten())

        peak = float(np.max(np.abs(audio)))

        print(f"Captured peak: {peak:.6f}")

        if peak < 0.0005:
            print("That was silence. The device is not capturing.")
            return None

        return normalize_audio(audio)

    except Exception as e:

        print("Push-to-talk error:", e)

        return None


# ============================================================
# RECORD SPEECH
# ============================================================

def record_speech():

    if PUSH_TO_TALK:
        return record_push_to_talk()

    ensure_input_device()

    print("\nListening...")

    noise_floor = calculate_noise_floor()

    # A dead stream reads ~0.0 here. Do not silently build a
    # threshold on top of a dead stream - say so.
    if noise_floor < 0.00005:

        print("WARNING: microphone appears to be returning silence.")
        print("Run: python voice_assistant.py --devices")
        print("Then set INPUT_DEVICE manually, or set PUSH_TO_TALK = True.")

    threshold = max(MIN_RMS_THRESHOLD, noise_floor * 2.5)

    print(f"Noise floor: {noise_floor:.6f}")
    print(f"Speech threshold: {threshold:.6f}")

    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    max_chunks = int(MAX_RECORD_SECONDS / CHUNK_DURATION)
    timeout_chunks = int(START_TIMEOUT / CHUNK_DURATION)
    silence_chunks_required = int(SILENCE_DURATION / CHUNK_DURATION)

    # --------------------------------------------------------
    # PRE-ROLL BUFFER
    # Keeps audio from just before speech detection so the
    # first word is not clipped.
    # --------------------------------------------------------

    pre_roll_chunks_required = max(
        1,
        int(PRE_ROLL_DURATION / CHUNK_DURATION)
    )

    pre_roll = []
    audio_chunks = []

    speech_started = False
    silence_count = 0
    chunks_waited = 0

    observed_peak = 0.0

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=chunk_samples,
            device=INPUT_DEVICE
        ) as stream:

            for _ in range(max_chunks):

                data, overflowed = stream.read(chunk_samples)

                chunk = apply_gain(data[:, 0].copy())

                rms = calculate_rms(chunk)

                observed_peak = max(observed_peak, rms)

                if overflowed:
                    print("Warning: microphone overflow.")

                # =============================================
                # BEFORE SPEECH
                # =============================================

                if not speech_started:

                    chunks_waited += 1

                    pre_roll.append(chunk)

                    if len(pre_roll) > pre_roll_chunks_required:
                        pre_roll.pop(0)

                    # Live meter so you can SEE the input level
                    if chunks_waited % 5 == 0:
                        print(
                            f"  level={rms:.6f} "
                            f"/ need {threshold:.6f}"
                        )

                    if rms >= threshold:

                        speech_started = True

                        print("Speech detected.")

                        audio_chunks.extend(pre_roll)

                    elif chunks_waited >= timeout_chunks:

                        print("No speech detected.")
                        print(f"Loudest level seen: {observed_peak:.6f}")

                        if observed_peak < 0.0005:
                            print("Mic is effectively silent -> device problem.")
                            print("Run with --devices and set INPUT_DEVICE.")
                        else:
                            print("Mic works but is quiet -> raise INPUT_GAIN.")

                        break

                # =============================================
                # SPEECH IN PROGRESS
                # =============================================

                else:

                    audio_chunks.append(chunk)

                    if rms < threshold:
                        silence_count += 1
                    else:
                        silence_count = 0

                    if silence_count >= silence_chunks_required:

                        print("Speech ended.")
                        break

    except Exception as e:

        print("Microphone error:", e)
        print("If this says 'Error opening InputStream', another app")
        print("is holding the mic, or the device index is wrong.")

        return None

    if not audio_chunks:
        return None

    audio = normalize_audio(
        np.concatenate(audio_chunks)
    )

    duration = len(audio) / SAMPLE_RATE

    print(f"Recorded: {duration:.2f} seconds")

    if duration < 0.35:
        return None

    return audio


# ============================================================
# WHISPER DEVICE
# ============================================================

def get_whisper_device():

    try:

        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"

    except Exception:
        pass

    return "cpu", "int8"


# ============================================================
# LOAD WHISPER
# ============================================================

_whisper_model = None


def get_whisper_model():

    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    device, compute_type = get_whisper_device()

    print("=" * 60)
    print("Loading Whisper...")
    print("Model:", WHISPER_MODEL)
    print("Device:", device)
    print("Compute:", compute_type)
    print("(First run downloads ~460 MB. This is not a hang.)")
    print("=" * 60)

    try:

        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type
        )

    except Exception as e:

        print("Whisper GPU initialization failed:")
        print(e)
        print("Falling back to CPU...")

        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )

    print("Whisper loaded successfully.")

    return _whisper_model


# ============================================================
# CLEAN TRANSCRIPT
# ============================================================

def clean_transcript(text):

    if not text:
        return ""

    text = str(text).strip()

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"([.!?,])\1{2,}", r"\1", text)

    return text.strip()


def normalize_command(text):
    """
    Strip punctuation for keyword matching.

    Whisper returns "exit." with a full stop, which is why the
    old exit-command list never matched.
    """

    if not text:
        return ""

    text = str(text).lower().strip()

    text = re.sub(r"[^a-z0-9 ]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# TRANSCRIPT VALIDATION
# ============================================================

def is_valid_transcript(text):

    if not text:
        return False

    text = text.strip()

    if len(text) < 2:
        return False

    if len(text) > 1000:
        return False

    words = text.lower().split()

    # --------------------------------------------------------
    # Repeated word protection
    # --------------------------------------------------------

    if len(words) >= 8:

        unique_words = set(words)

        repetition_ratio = len(unique_words) / len(words)

        if repetition_ratio < 0.25:
            return False

    # --------------------------------------------------------
    # Repeated phrase protection
    # --------------------------------------------------------

    if len(words) >= 12:

        for phrase_size in [2, 3, 4]:

            counts = {}

            for i in range(len(words) - phrase_size + 1):

                phrase = " ".join(words[i:i + phrase_size])

                counts[phrase] = counts.get(phrase, 0) + 1

            if counts and max(counts.values()) >= 4:
                return False

    # --------------------------------------------------------
    # Known Whisper hallucination phrases
    # --------------------------------------------------------

    suspicious_phrases = [
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "subscribe to my channel",
        "see you in the next video",
        "subtitles by",
        "amara.org",
        "you"
    ]

    lowered = normalize_command(text)

    for phrase in suspicious_phrases:

        if lowered == normalize_command(phrase):
            return False

    return True


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(audio):

    if audio is None:
        return ""

    if len(audio) == 0:
        return ""

    model = get_whisper_model()

    try:

        segments, info = model.transcribe(
            audio,
            language="en",

            # Accuracy
            beam_size=3,
            best_of=3,
            temperature=0.0,

            # Stop previous text from influencing transcription
            condition_on_previous_text=False,

            # Hallucination protection
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,

            # VAD loosened. The old 500ms / 400ms settings could
            # delete short commands like "open youtube" entirely.
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200
            },
            initial_prompt=INITIAL_PROMPT
        )

        texts = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            avg_logprob = getattr(segment, "avg_logprob", 0.0)
            no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
            compression_ratio = getattr(segment, "compression_ratio", 0.0)

            if no_speech_prob >= 0.85:
                continue

            if avg_logprob < -1.5:
                continue

            if compression_ratio > 3.0:
                continue

            texts.append(text)

        if not texts:
            print("Whisper found no usable speech in the clip.")
            return ""

        transcript = clean_transcript(" ".join(texts))

        if not is_valid_transcript(transcript):

            print("Suspicious transcript rejected:")
            print(transcript)

            return ""

        detected_language = getattr(info, "language", "unknown")

        print("Detected language:", detected_language)
        print("You:", transcript)

        return transcript.lower()

    except Exception as e:

        print("Whisper error:", e)

        return ""


# ============================================================
# MAIN LISTEN FUNCTION
# ============================================================

def listen():

    try:

        audio = record_speech()

        if audio is None:
            print("No speech captured.")
            return ""

        transcript = transcribe_audio(audio)

        if not transcript:
            print("Could not understand speech.")
            return ""

        return transcript

    except KeyboardInterrupt:
        return ""

    except Exception as e:

        print("Listen error:", e)

        return ""


# ============================================================
# REMINDER
# ============================================================

def set_reminder(message, seconds):

    def reminder():

        time.sleep(seconds)

        # speak() is lock-protected, so this is safe now.
        speak(f"Reminder: {message}")

    threading.Thread(
        target=reminder,
        daemon=True
    ).start()


# ============================================================
# OPEN URL IN NEW CHROME TAB
# ============================================================

CHROME_PATHS = [

    r"C:\Program Files\Google\Chrome\Application\chrome.exe",

    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",

    os.path.expandvars(
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
    ),

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",

    "/usr/bin/google-chrome",
]


def find_chrome():

    for path in CHROME_PATHS:

        try:
            if path and os.path.exists(path):
                return path
        except Exception:
            continue

    return None


def open_url_in_new_chrome_tab(url):
    """
    Opens `url` in a NEW tab.

    Uses Chrome with --new-tab when available, otherwise the
    system default browser.
    """

    chrome_path = find_chrome()

    if chrome_path:

        try:

            subprocess.Popen([chrome_path, "--new-tab", url])

            return True

        except Exception as e:

            print("Chrome launch failed, using default browser:", e)

    webbrowser.open_new_tab(url)

    return True


# ============================================================
# SAFE WEB SEARCH
# ============================================================
# Fixes four failure modes:
#   1. duckduckgo_search is deprecated -> constant rate limits
#   2. a single failed call returned nothing with no retry
#   3. only one backend was used; when it throttles all dies
#   4. result keys differ by version (href / link / url)
#
# Always returns a list of {"title", "url", "body"} dicts,
# never raises.
# ============================================================

SEARCH_BACKENDS = [
    "duckduckgo",
    "bing",
    "brave",
    "mojeek",
    "yahoo",
]


def _normalise_result(result):

    if not isinstance(result, dict):
        return None

    url = (
        result.get("href")
        or result.get("url")
        or result.get("link")
        or ""
    ).strip()

    title = (
        result.get("title")
        or result.get("name")
        or ""
    ).strip()

    body = (
        result.get("body")
        or result.get("snippet")
        or result.get("description")
        or ""
    ).strip()

    if not url:
        return None

    if not url.startswith(("http://", "https://")):
        return None

    return {
        "title": title or url,
        "url": url,
        "body": body
    }


def web_search(query, max_results=5, region="in-en", retries=2):

    if DDGS is None:
        print("Search unavailable: run  pip install -U ddgs")
        return []

    query = (query or "").strip()

    if not query:
        return []

    for backend in SEARCH_BACKENDS:

        for attempt in range(retries + 1):

            try:

                with DDGS() as ddgs:

                    try:

                        raw = ddgs.text(
                            query,
                            region=region,
                            safesearch="moderate",
                            max_results=max_results,
                            backend=backend
                        )

                    except TypeError:

                        # Legacy duckduckgo_search has no `backend`
                        raw = ddgs.text(
                            query,
                            region=region,
                            safesearch="moderate",
                            max_results=max_results
                        )

                    results = []

                    for item in (raw or []):

                        clean = _normalise_result(item)

                        if clean:
                            results.append(clean)

                if results:

                    print(
                        f"Search OK via '{backend}': "
                        f"{len(results)} results"
                    )

                    return results

                # An empty response usually means soft throttling
                print(
                    f"Backend '{backend}' returned nothing "
                    f"(attempt {attempt + 1})"
                )

            except _DDGS_ERRORS as e:

                print(
                    f"Backend '{backend}' error "
                    f"(attempt {attempt + 1}): {e}"
                )

            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

        time.sleep(0.5)

    print("All search backends failed for:", query)

    return []


# ============================================================
# KNOWN WEBSITES
# ============================================================
# Direct name -> URL map. Opens instantly, no web search, more
# reliable than trusting whatever a search returns first.
# ============================================================

KNOWN_WEBSITES = {
    "youtube": "https://www.youtube.com",
    "github": "https://github.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "linkedin": "https://www.linkedin.com",
    "hugging face": "https://huggingface.co",
    "nvidia": "https://www.nvidia.com",
    "microsoft": "https://www.microsoft.com",
    "stackoverflow": "https://stackoverflow.com",
    "tensorflow": "https://www.tensorflow.org",
    "python": "https://www.python.org",
    "mongodb": "https://www.mongodb.com",
    "oracle": "https://www.oracle.com",
    "amazon": "https://www.amazon.com",
    "netflix": "https://www.netflix.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "whatsapp": "https://web.whatsapp.com",
    "chatgpt": "https://chatgpt.com",
    "naukri": "https://www.naukri.com",
    "flipkart": "https://www.flipkart.com",
    "leetcode": "https://leetcode.com",
    "geeksforgeeks": "https://www.geeksforgeeks.org",
    "wikipedia": "https://www.wikipedia.org",
    "reddit": "https://www.reddit.com",
    "spotify": "https://open.spotify.com",
}


# Matches  example.com  /  www.example.co.in  /  site.com/page
DOMAIN_PATTERN = re.compile(
    r"^(https?://)?([a-z0-9-]+\.)+[a-z]{2,}(/\S*)?$"
)


def resolve_website_name(name):

    name = (name or "").lower().strip()

    if not name:
        return name

    best_match = None
    best_score = 0

    for website in KNOWN_WEBSITES:

        score = fuzz.ratio(name, website)

        if score > best_score:

            best_score = score
            best_match = website

    print(f"Website match: {name} -> {best_match} ({best_score})")

    if best_score >= 70:
        return best_match

    return name


# ============================================================
# SEARCH AND OPEN ANY WEBSITE
# ============================================================

def search_and_open_website(website_name, announce=True):
    """
    Open a website in a NEW TAB.

    Resolution order:
      1. Full domain / URL   (example.com)  -> opened directly
      2. Known website map   (youtube, ...) -> opened directly
      3. Anything else                      -> web search

    announce=True speaks through pyttsx3 (CLI use).
    announce=False lets the caller handle output (Streamlit).

    Returns: (success, message)
    """

    def respond(message, success):

        if announce and message:
            speak(message)

        return success, message

    website_name = (website_name or "").strip().lower()

    if not website_name:
        return respond("Please tell me the website name.", False)

    # ----------------------------------------------------
    # DIRECT DOMAIN / URL
    # ----------------------------------------------------

    if DOMAIN_PATTERN.match(website_name):

        url = website_name

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        print("\n" + "=" * 60)
        print("DOMAIN - OPENING DIRECTLY")
        print("URL:", url)
        print("=" * 60)

        open_url_in_new_chrome_tab(url)

        update_context(
            command=f"open {website_name}",
            intent="open_website",
            website=website_name
        )

        return respond(
            f"Opening {website_name} in a new tab.",
            True
        )

    resolved_name = resolve_website_name(website_name)

    if not resolved_name:
        return respond("Please tell me the website name.", False)

    # ----------------------------------------------------
    # FAST PATH - known website
    # ----------------------------------------------------

    if resolved_name in KNOWN_WEBSITES:

        selected_url = KNOWN_WEBSITES[resolved_name]

        print("\n" + "=" * 60)
        print("KNOWN WEBSITE - OPENING DIRECTLY")
        print("URL:", selected_url)
        print("=" * 60)

        open_url_in_new_chrome_tab(selected_url)

        update_context(
            command=f"open {resolved_name}",
            intent="open_website",
            website=resolved_name
        )

        return respond(
            f"Opening {resolved_name} in a new tab.",
            True
        )

    # ----------------------------------------------------
    # FALLBACK - search for it
    # ----------------------------------------------------

    website_name = resolved_name

    print("\n" + "=" * 60)
    print("WEBSITE SEARCH")
    print("=" * 60)
    print("Searching for:", website_name)

    results = web_search(
        f"{website_name} official website",
        max_results=5
    )

    # ----------------------------------------------------
    # LAST RESORT
    # If every backend is throttled, guess the likeliest
    # domain. A wrong guess still beats "I could not find it".
    # ----------------------------------------------------

    if not results:

        guess = re.sub(r"[^a-z0-9]", "", website_name)

        if guess:

            guessed_url = f"https://www.{guess}.com"

            print("Search failed - trying guessed URL:", guessed_url)

            open_url_in_new_chrome_tab(guessed_url)

            update_context(
                command=f"open {website_name}",
                intent="open_website",
                website=website_name
            )

            return respond(
                f"Search was unavailable, so I opened "
                f"{guess} dot com in a new tab.",
                True
            )

        return respond(
            f"I could not find the website {website_name}.",
            False
        )

    # ----------------------------------------------------
    # BEST RESULT
    # ----------------------------------------------------

    best = results[0]

    print("Website title:", best["title"])
    print("Website URL:", best["url"])

    open_url_in_new_chrome_tab(best["url"])

    update_context(
        command=f"open {website_name}",
        intent="open_website",
        website=website_name
    )

    print("=" * 60)
    print("WEBSITE OPENED IN NEW TAB")
    print("=" * 60)

    return respond(
        f"Opening {website_name} in a new tab.",
        True
    )


# ============================================================
# UPLOADED DOCUMENT VECTORSTORE (cached, standalone CLI use)
# ============================================================
# When this file runs on its own there is no Streamlit session,
# so we load whatever vectorstore rag_engine last saved to disk.
# When imported from app.py, the caller passes the LIVE session
# FAISS db into voice_answer() and this cache is unused.
# ============================================================

_cached_db = None
_db_loaded = False


def get_document_db():

    global _cached_db, _db_loaded

    if not _db_loaded:

        _db_loaded = True

        rag = get_rag_engine()

        if rag is None:
            _cached_db = None
            return None

        try:
            _cached_db = rag.load_vectorstore()
        except Exception as e:
            print("Could not load saved vectorstore:", e)
            _cached_db = None

    return _cached_db


# ============================================================
# RAG + WEB ANSWER  (documents first, web fallback, WITH sources)
# ============================================================

def describe_source(source_type):

    if source_type == "document":
        return "Yeh jankari maine aapke uploaded document se li hai."

    if source_type == "web":
        return "Yeh jankari maine web search se li hai."

    if source_type == "both":
        return "Yeh jawaab aapke document aur web search dono se mila hai."

    return ""


def format_document_sources(doc_sources, limit=3):

    if not doc_sources:
        return ""

    seen = set()
    lines = []

    for doc in doc_sources:

        metadata = getattr(doc, "metadata", None) or {}

        filename = metadata.get(
            "filename",
            metadata.get("source", "Unknown file")
        )

        page = metadata.get("page")
        slide = metadata.get("slide")
        sheet = metadata.get("sheet")

        if page is not None:
            try:
                location = f"Page {int(page) + 1}"
            except (ValueError, TypeError):
                location = f"Page {page}"
        elif slide is not None:
            location = f"Slide {slide}"
        elif sheet is not None:
            location = f"Sheet: {sheet}"
        else:
            location = "Document"

        key = (str(filename), location)

        if key in seen:
            continue

        seen.add(key)

        lines.append(
            f"- {os.path.basename(str(filename))} ({location})"
        )

        if len(lines) >= limit:
            break

    return "\n".join(lines)


def format_web_sources(web_sources, limit=3):

    if not web_sources:
        return ""

    lines = []

    for index, source in enumerate(web_sources[:limit], start=1):

        if not isinstance(source, dict):
            continue

        title = source.get("title", "Unknown source")

        url = (
            source.get("url")
            or source.get("href")
            or source.get("link")
            or ""
        )

        if url:
            lines.append(f"{index}. {title} - {url}")
        else:
            lines.append(f"{index}. {title}")

    return "\n".join(lines)


def voice_answer(question, db=None, context=None):
    """
    Single entry point used by BOTH the Streamlit app and this
    standalone CLI assistant.

    Returns: answer_text, reference_note, sources, source_type
    """

    rag = get_rag_engine()

    if rag is None:

        return (
            "Document engine load nahi hua. "
            f"Reason: {_rag_import_error}",
            "",
            [],
            "error"
        )

    RateLimit = get_rate_limit_error()

    try:

        answer, sources, source_type = rag.answer_question(question, db)

    except RateLimit as e:

        return (
            f"Groq ka daily token limit khatam ho gaya hai. {e}",
            "",
            [],
            "error"
        )

    except Exception as e:

        return (
            f"Sorry, ek error aa gaya: {e}",
            "",
            [],
            "error"
        )

    reference_note = describe_source(source_type)

    if source_type == "document":

        docs_text = format_document_sources(sources)

        if docs_text:
            reference_note += "\n" + docs_text

    elif source_type == "web":

        web_text = format_web_sources(sources)

        if web_text:
            reference_note += "\n" + web_text

    elif source_type == "both" and isinstance(sources, dict):

        docs_text = format_document_sources(sources.get("documents", []))
        web_text = format_web_sources(sources.get("web", []))

        if docs_text:
            reference_note += "\nDocuments:\n" + docs_text

        if web_text:
            reference_note += "\nWeb:\n" + web_text

    return answer, reference_note.strip(), sources, source_type


# ============================================================
# "OPEN ..." COMMAND PARSING
# ============================================================
# Understands English + Hinglish:
#   open youtube          open youtube in a new tab
#   launch github         go to linkedin       visit naukri
#   youtube kholo         google khol do       gmail open karo
#   open flipkart dot com open example.com
# ============================================================

OPEN_PREFIXES = (
    "open ",
    "launch ",
    "go to ",
    "goto ",
    "visit ",
)

OPEN_SUFFIXES = (
    " kholo",
    " khol do",
    " khol de",
    " kholiye",
    " khol dijiye",
    " open karo",
    " open kar do",
    " open kar de",
    " open kijiye",
)

MAX_OPEN_TARGET_WORDS = 4


def extract_open_target(command):
    """
    Returns what the user wants opened ("youtube"), or None if
    this is not an "open ..." style command.
    """

    target = None

    for prefix in OPEN_PREFIXES:

        if command.startswith(prefix):

            target = command[len(prefix):]
            break

    if target is None:

        for suffix in OPEN_SUFFIXES:

            if command.endswith(suffix):

                target = command[:-len(suffix)]
                break

    if target is None:
        return None

    # "... in a new tab" / "... on new chrome tab"
    target = re.sub(
        r"\b(in|on|into)\s+(a\s+|the\s+)?new\s+(chrome\s+)?(tab|window)\b",
        " ",
        target
    )

    # "... in chrome" / "... on google chrome"
    target = re.sub(
        r"\b(in|on)\s+(google\s+)?chrome\b",
        " ",
        target
    )

    # "flipkart dot com" -> "flipkart.com"
    target = re.sub(r"\s+dot\s+", ".", target)

    # Filler words
    target = re.sub(r"^(the|a|my)\s+", "", target.strip())

    target = re.sub(
        r"\s+(website|site|webpage|web page|please|for me|na|yaar)$",
        "",
        target.strip()
    )

    return re.sub(r"\s+", " ", target).strip()


def _matches(command, exact=(), starts=()):

    if command in exact:
        return True

    return any(command.startswith(s) for s in starts)


# ============================================================
# SYSTEM COMMAND HANDLER
# ============================================================

VS_CODE_PATHS = [

    os.path.expandvars(
        r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"
    ),

    r"C:\Program Files\Microsoft VS Code\Code.exe",

    "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",

    "/usr/bin/code",
]


def try_handle_system_command(command, announce=True):
    """
    Detect and execute a non-RAG "system" command: opening an
    app or website in a NEW TAB, telling the time/date, setting
    a reminder, or doing an explicit web search.

    Used by the CLI loop, the Streamlit voice assistant, and
    the Streamlit text chat.

    Returns: (handled, message)
        handled=False means "treat this as a normal question".
    """

    def respond(message, handled=True):

        if announce and message:
            speak(message)

        return handled, message

    if not command:
        return False, ""

    # --------------------------------------------------------
    # CLEAN COMMAND
    # --------------------------------------------------------

    command = command.strip().lower()

    # A dot is only removed when NOT between two word chars, so
    # "open flipkart.com" survives intact.
    command = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", command)

    command = re.sub(r"[!?,]+", " ", command)

    command = re.sub(r"\s+", " ", command).strip()

    print("\nProcessed command:", command)

    # ========================================================
    # OPEN APP / WEBSITE
    # ========================================================

    target = extract_open_target(command)

    if target:

        # ----------------------------------------------------
        # Desktop apps
        # ----------------------------------------------------

        if target in ("chrome", "google chrome"):

            chrome_path = find_chrome()

            if chrome_path:

                subprocess.Popen([chrome_path])

                return respond("Opening Google Chrome.")

            webbrowser.open_new_tab("https://www.google.com")

            return respond("Opening Chrome.")

        if target in (
            "visual studio code",
            "vs code",
            "vscode",
            "code"
        ):

            for path in VS_CODE_PATHS:

                if path and os.path.exists(path):

                    subprocess.Popen([path])

                    return respond("Opening Visual Studio Code.")

            return respond("Visual Studio Code was not found.")

        if target == "notepad":

            try:
                subprocess.Popen("notepad.exe")
                return respond("Opening Notepad.")
            except Exception:
                return respond("Notepad is not available on this system.")

        if target in ("calculator", "calc"):

            try:
                subprocess.Popen("calc.exe")
                return respond("Opening Calculator.")
            except Exception:
                return respond("Calculator is not available on this system.")

        # ----------------------------------------------------
        # Any website -> NEW TAB
        #
        # Guard: a long sentence such as
        # "open source tools for machine learning kya hain"
        # is a QUESTION, not a website. Leave it to RAG.
        # ----------------------------------------------------

        if len(target.split()) <= MAX_OPEN_TARGET_WORDS:

            print("Website command detected:", target)

            _success, message = search_and_open_website(
                target,
                announce=False
            )

            return respond(message)

    # ========================================================
    # TIME
    # ========================================================

    if _matches(
        command,
        exact=(
            "time",
            "current time",
            "the time",
            "time batao",
            "samay batao",
            "abhi kya time hua hai",
        ),
        starts=(
            "what time is it",
            "what is the time",
            "what's the time",
            "whats the time",
            "tell me the time",
            "what is the current time",
            "what's the current time",
        )
    ):

        now = datetime.datetime.now().strftime("%I:%M %p")

        update_context(command, intent="time")

        return respond(f"The time is {now}")

    # ========================================================
    # DATE
    # ========================================================

    if _matches(
        command,
        exact=(
            "date",
            "today's date",
            "todays date",
            "today date",
            "aaj ki date batao",
            "aaj ki tarikh batao",
        ),
        starts=(
            "what date is it",
            "what is the date",
            "what's the date",
            "whats the date",
            "what is today's date",
            "what's today's date",
            "what is todays date",
            "tell me the date",
            "tell me today's date",
        )
    ):

        today = datetime.datetime.now().strftime("%d %B %Y")

        update_context(command, intent="date")

        return respond(f"Today is {today}")

    # ========================================================
    # REMINDER
    # ========================================================

    if command.startswith("remind me to"):

        try:

            task = command.split("remind me to", 1)[1].strip()

            for unit, seconds_per in (("minute", 60), ("hour", 3600)):

                if " in " in task and unit in task:

                    message = task.split(" in ", 1)[0].strip()

                    time_part = task.split(" in ", 1)[1]

                    amount = int(time_part.split()[0])

                    update_context(command, intent="reminder")

                    set_reminder(message, amount * seconds_per)

                    return respond(
                        f"Reminder set. I will remind you to "
                        f"{message} in {amount} {unit}s."
                    )

            return respond(
                "Please specify the reminder time in minutes or hours."
            )

        except Exception as e:

            print("Reminder error:", e)

            return respond("I could not understand the reminder time.")

    # ========================================================
    # EXPLICIT WEB SEARCH
    # ========================================================

    SEARCH_PREFIXES = (
        "search for ",
        "search ",
        "google karo ",
        "search karo ",
        "web search ",
        "google ",
    )

    matched_prefix = None

    for prefix in SEARCH_PREFIXES:

        if command.startswith(prefix):

            matched_prefix = prefix
            break

    if matched_prefix:

        query = command[len(matched_prefix):].strip()

        query = re.sub(
            r"\s+(karo|kar do|kijiye|please)$",
            "",
            query
        ).strip()

        if not query:
            return respond("What should I search for?")

        update_context(command, intent="web_search", topic=query)

        answer = None
        web_sources = []

        rag = get_rag_engine()

        if rag is not None:

            RateLimit = get_rate_limit_error()

            try:

                answer, web_sources = rag.answer_from_web(query)

            except RateLimit as e:

                return respond(
                    f"Groq ka daily token limit khatam ho gaya hai. {e}"
                )

            except Exception as e:

                print("rag_engine web search failed:", e)
                answer = None

        # Fallback: raw results from our own safe searcher
        if not answer:

            web_sources = web_search(query, max_results=5)

            if not web_sources:

                return respond(
                    "Web search abhi available nahi hai. "
                    "Thodi der baad try kijiye."
                )

            answer = (
                web_sources[0].get("body")
                or "Yeh top results mile:"
            )

        sources_text = format_web_sources(web_sources)

        full_message = answer

        if sources_text:
            full_message += "\n\n" + sources_text

        return respond(full_message)

    # ========================================================
    # NOT A SYSTEM COMMAND
    # ========================================================

    return False, ""


# ============================================================
# PROCESS COMMAND  (standalone CLI loop)
# ============================================================

def process_command(command):

    if not command:
        return

    handled, _message = try_handle_system_command(
        command,
        announce=True
    )

    if handled:
        return

    # ========================================================
    # DOCUMENT / WEB / GENERAL QUESTION
    # ========================================================

    db = get_document_db()

    answer, reference_note, _sources, source_type = voice_answer(
        command,
        db,
        conversation_context
    )

    update_context(command=command, intent="question")

    speak(answer)

    if source_type != "error" and reference_note:
        print("\n" + reference_note)


# ============================================================
# DIAGNOSTICS
# ============================================================

def test_microphone():

    print("=" * 60)
    print("MICROPHONE TEST")
    print("=" * 60)

    audio = record_speech()

    if audio is None:
        print("No audio captured.")
        return

    text = transcribe_audio(audio)

    print("=" * 60)
    print("FINAL TRANSCRIPT")
    print("=" * 60)
    print(text)


def test_search(query="python official website"):

    print("=" * 60)
    print("WEB SEARCH TEST")
    print("Query:", query)
    print("=" * 60)

    results = web_search(query, max_results=5)

    if not results:
        print("NO RESULTS - every backend failed or is rate limited.")
        return

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title']}")
        print("   ", result["url"])


def run_diagnostics():

    print("=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)

    print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))

    rag = get_rag_engine()
    print("rag_engine imported:", rag is not None)

    print("Search package:", "ok" if DDGS else "MISSING (pip install -U ddgs)")

    print("Chrome found:", find_chrome() or "no")

    print("-" * 60)

    print_input_devices()

    ensure_input_device(force_reprobe=True)


# ============================================================
# MAIN
# ============================================================

EXIT_COMMANDS = {
    "exit",
    "quit",
    "stop",
    "goodbye",
    "good bye",
    "close assistant",
    "band karo",
}


def main():

    # Fail fast on a dead microphone instead of looping forever.
    if not PUSH_TO_TALK:

        if ensure_input_device() is None:

            print()
            print("No working microphone. Options:")
            print("  1. Fix the OS issue listed above, then rerun.")
            print("  2. Set PUSH_TO_TALK = True in this file.")
            print("  3. Set INPUT_DEVICE = <index> from --devices.")
            print()

    speak(
        "Hello, I am your AI Research Assistant. "
        "Ask me anything."
    )

    consecutive_failures = 0

    while True:

        try:
            question = listen()
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        if not question:

            consecutive_failures += 1

            if consecutive_failures >= 3:

                print()
                print("Three failed captures in a row.")
                print("Run: python voice_assistant.py --devices")
                print("Or set PUSH_TO_TALK = True.")
                print()

                consecutive_failures = 0

            speak("Sorry, I could not understand you.")

            continue

        consecutive_failures = 0

        if normalize_command(question) in EXIT_COMMANDS:

            speak("Goodbye. Have a great day.")
            break

        try:
            process_command(question)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print("Command error:", e)
            speak("Sorry, something went wrong.")


if __name__ == "__main__":

    args = [a.lower() for a in sys.argv[1:]]

    if "--devices" in args:
        print_input_devices()
        ensure_input_device(force_reprobe=True)

    elif "--mic" in args:
        test_microphone()

    elif "--search" in args:
        test_search()

    elif "--doctor" in args:
        run_diagnostics()

    else:
        main()