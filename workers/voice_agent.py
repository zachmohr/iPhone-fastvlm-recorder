#!/usr/bin/env python3
"""Consume topic questions and synthesize spoken prompts via OpenAI TTS."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import redis
from openai import OpenAI

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
TOPIC_STREAM = os.environ.get("REDIS_TOPIC_STREAM", "iphonelog:topics")
GROUP = os.environ.get("VOICE_CONSUMER_GROUP", "voice-agent")
CONSUMER = os.environ.get("REDIS_CONSUMER_NAME", f"voice-{os.getpid()}")
VOICE_MODEL = os.environ.get("VOICE_MODEL", "gpt-4o-mini-tts")
VOICE_NAME = os.environ.get("VOICE_VOICE", "alloy")
LOG_DIR = Path(os.environ.get("ORCHESTRATOR_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "voice_questions.jsonl"
AUDIO_DIR = LOG_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MAX_STREAM_LENGTH = int(os.environ.get("VOICE_STREAM_MAXLEN", "500"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY must be set for voice agent.")

client = OpenAI(api_key=OPENAI_API_KEY)


def setup_redis_client() -> redis.Redis:
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    client.ping()
    try:
        client.xgroup_create(name=TOPIC_STREAM, groupname=GROUP, id="0", mkstream=True)
        print(f"[VoiceAgent] Created consumer group '{GROUP}' on stream '{TOPIC_STREAM}'.")
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    return client


def append_log(record: dict):
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def synthesize_audio(text: str) -> Path | None:
    if not text:
        return None
    try:
        response = client.audio.speech.create(
            model=VOICE_MODEL,
            voice=VOICE_NAME,
            input=text,
            response_format="wav",
        )
    except Exception as exc:
        print(f"[VoiceAgent] TTS call failed: {exc}", flush=True)
        return None

    if hasattr(response, "read"):
        audio_bytes = response.read()
    elif isinstance(response, (bytes, bytearray)):
        audio_bytes = bytes(response)
    else:
        audio_bytes = getattr(response, "audio", None)
        if audio_bytes is None:
            print("[VoiceAgent] Unexpected TTS response format; no audio payload.", flush=True)
            return None
        if isinstance(audio_bytes, str):
            import base64
            audio_bytes = base64.b64decode(audio_bytes)

    with tempfile.NamedTemporaryFile(
        prefix="voice_question_", suffix=".wav", dir=AUDIO_DIR, delete=False
    ) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        path = Path(tmp.name)
        print(f"[VoiceAgent] Saved TTS audio to {path}", flush=True)
        return path


def play_audio(path: Path):
    try:
        subprocess.run(["afplay", str(path)], check=True)
    except Exception as exc:
        print(f"[VoiceAgent] Failed to play audio {path}: {exc}")


def process_event(rclient: redis.Redis, message_id: str, payload: dict):
    try:
        data = json.loads(payload.get("data", "{}"))
    except json.JSONDecodeError:
        rclient.xack(TOPIC_STREAM, GROUP, message_id)
        return

    question = data.get("question", "").strip()
    if not question:
        rclient.xack(TOPIC_STREAM, GROUP, message_id)
        return

    audio_path = synthesize_audio(question)
    record = {
        "session_id": data.get("session_id"),
        "topic": data.get("topic"),
        "question": question,
        "summary": data.get("summary"),
        "generated_ts": data.get("generated_ts", time.time()),
        "audio_path": str(audio_path) if audio_path else "",
    }
    append_log(record)

    if audio_path:
        play_audio(audio_path)

    rclient.xack(TOPIC_STREAM, GROUP, message_id)


def main():
    rclient = setup_redis_client()
    print(f"[VoiceAgent] Listening on '{TOPIC_STREAM}' as '{CONSUMER}'")

    running = True

    def _handle_signal(signum, *_):
        nonlocal running
        running = False
        print(f"[VoiceAgent] Received signal {signum}. Shutting down...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while running:
        try:
            response = rclient.xreadgroup(
                GROUP,
                CONSUMER,
                {TOPIC_STREAM: ">"},
                count=5,
                block=5000,
            )
            if not response:
                continue
            for _, messages in response:
                for message_id, payload in messages:
                    process_event(rclient, message_id, payload)
        except redis.exceptions.ConnectionError as exc:
            print(f"[VoiceAgent] Redis connection lost: {exc}. Retrying in 5s...")
            time.sleep(5)
            rclient = setup_redis_client()
        except Exception as exc:
            print(f"[VoiceAgent] Error: {exc}")
            time.sleep(1)

    print("[VoiceAgent] Shutdown complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[VoiceAgent] Interrupted.")
