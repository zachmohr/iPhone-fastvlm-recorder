#!/usr/bin/env python3
"""Redis-backed orchestrator that summarizes caption events via a local Ollama model."""

from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import redis


BACKEND = os.environ.get("ORCHESTRATOR_BACKEND", "ollama").strip().lower()

ollama = None
openai_client = None

if BACKEND == "ollama":
    try:
        import ollama  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - surfaced on startup
        raise SystemExit(
            "Install the 'ollama' package inside this environment or set ORCHESTRATOR_BACKEND=openai."
        ) from exc
elif BACKEND in {"openai", "api"}:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - surfaced on startup
        raise SystemExit(
            "Install the 'openai' package (pip install openai) or switch ORCHESTRATOR_BACKEND back to ollama."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required when ORCHESTRATOR_BACKEND=openai")
    openai_client = OpenAI(api_key=api_key)
else:  # pragma: no cover - configuration guard
    raise SystemExit(f"Unsupported ORCHESTRATOR_BACKEND '{BACKEND}'. Use 'ollama' or 'openai'.")


REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
CAPTION_STREAM = os.environ.get("REDIS_CAPTION_STREAM", "iphonelog:captions")
INSIGHT_STREAM = os.environ.get("REDIS_INSIGHT_STREAM", "iphonelog:insights")
GROUP = os.environ.get("REDIS_CONSUMER_GROUP", "orchestrator")
CONSUMER = os.environ.get("REDIS_CONSUMER_NAME", f"orch-{os.getpid()}")

if BACKEND == "ollama":
    MODEL = os.environ.get("ORCHESTRATOR_MODEL", "llama3")
else:
    MODEL = os.environ.get("ORCHESTRATOR_MODEL", "gpt-4o-mini")
MAX_STREAM_LENGTH = int(os.environ.get("INSIGHT_STREAM_MAXLEN", "1000"))
LOG_DIR = Path(os.environ.get("ORCHESTRATOR_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "orchestrator_insights.jsonl"


def setup_redis_client() -> redis.Redis:
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    client.ping()
    try:
        client.xgroup_create(name=CAPTION_STREAM, groupname=GROUP, id="0", mkstream=True)
        print(f"[Orchestrator] Created consumer group '{GROUP}' on stream '{CAPTION_STREAM}'.")
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    return client


def build_prompt(event: dict) -> list[dict[str, str]]:
    meta = event.get("meta", {})
    context_lines = [
        f"VLM text: {event.get('vlm_text', '').strip()}",
        f"HUD caption: {event.get('caption', '').strip()}",
        f"Faces detected: {meta.get('faces')} | Hands: {meta.get('hand_count')} | Gestures: {meta.get('gestures')}"
    ]

    user_payload = "\n".join(context_lines)
    return [
        {
            "role": "system",
            "content": (
                "You triage live scene captions. Identify noteworthy observations, uncertainties, or follow-up "
                "questions worth asking the person being recorded. Respond with concise bullet points."
            ),
        },
        {
            "role": "user",
            "content": (
                "Session: {session}\nTimestamp: {ts}\n{payload}".format(
                    session=event.get("session_id", "unknown"),
                    ts=datetime.fromtimestamp(event.get("timestamp", time.time())).isoformat(),
                    payload=user_payload,
                )
            ),
        },
    ]


def write_output(record: dict):
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_event(client: redis.Redis, message_id: str, raw_event: dict):
    try:
        event = json.loads(raw_event.get("data", "{}"))
    except json.JSONDecodeError:
        print(f"[Orchestrator] Skipping malformed payload: {raw_event}")
        client.xack(CAPTION_STREAM, GROUP, message_id)
        return

    if not event.get("vlm_text"):
        client.xack(CAPTION_STREAM, GROUP, message_id)
        return

    prompts = build_prompt(event)
    try:
        if BACKEND == "ollama":
            response = ollama.chat(model=MODEL, messages=prompts)
            content = response.get("message", {}).get("content", "").strip()
        else:
            response = openai_client.chat.completions.create(  # type: ignore[union-attr]
                model=MODEL,
                messages=prompts,
            )
            choices = getattr(response, "choices", [])
            content = choices[0].message.content.strip() if choices else ""
    except Exception as exc:
        print(f"[Orchestrator] Inference call failed: {exc}")
        time.sleep(2.0)
        return  # Keep message pending so we can retry later

    if not content:
        client.xack(CAPTION_STREAM, GROUP, message_id)
        return

    record = {
        "session_id": event.get("session_id"),
        "caption_ts": event.get("timestamp"),
        "insight_ts": time.time(),
        "vlm_text": event.get("vlm_text"),
        "hud_caption": event.get("caption"),
        "insights": content,
    }
    write_output(record)

    client.xadd(
        INSIGHT_STREAM,
        {"data": json.dumps(record, ensure_ascii=False)},
        maxlen=MAX_STREAM_LENGTH,
        approximate=True,
    )
    client.xack(CAPTION_STREAM, GROUP, message_id)
    print(f"[Orchestrator] Processed {message_id} -> {len(content.splitlines())} lines")


def main():
    client = setup_redis_client()
    print(
        f"[Orchestrator] Listening on stream '{CAPTION_STREAM}' as consumer '{CONSUMER}' with model '{MODEL}'"
        f" (backend={BACKEND})."
    )

    running = True

    def _handle_signal(signum, *_):
        nonlocal running
        running = False
        print(f"[Orchestrator] Received signal {signum}. Shutting down...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while running:
        try:
            response = client.xreadgroup(
                GROUP,
                CONSUMER,
                {CAPTION_STREAM: ">"},
                count=5,
                block=5000,
            )
            if not response:
                continue
            for _, messages in response:
                for message_id, payload in messages:
                    process_event(client, message_id, payload)
        except redis.exceptions.ConnectionError as exc:
            print(f"[Orchestrator] Lost Redis connection: {exc}. Retrying in 5s...")
            time.sleep(5)
            client = setup_redis_client()
        except Exception as exc:
            print(f"[Orchestrator] Error: {exc}")
            time.sleep(1)

    print("[Orchestrator] Shutdown complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Orchestrator] Interrupted.")
