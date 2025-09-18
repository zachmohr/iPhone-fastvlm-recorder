#!/usr/bin/env python3
"""Aggregate orchestrator insights into topic summaries and suggested questions."""

from __future__ import annotations

import json
import os
import signal
import time
from collections import deque
from pathlib import Path

import redis
from openai import OpenAI

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
INSIGHT_STREAM = os.environ.get("REDIS_INSIGHT_STREAM", "iphonelog:insights")
TOPIC_STREAM = os.environ.get("REDIS_TOPIC_STREAM", "iphonelog:topics")
GROUP = os.environ.get("REDIS_TOPIC_GROUP", "topic-aggregator")
CONSUMER = os.environ.get("REDIS_TOPIC_CONSUMER", f"topic-{os.getpid()}")
BUFFER_SIZE = int(os.environ.get("TOPIC_BUFFER_SIZE", "5"))
TIME_WINDOW = int(os.environ.get("TOPIC_TIME_WINDOW", "30"))
COOLDOWN = int(os.environ.get("TOPIC_COOLDOWN", "600"))
REASONING_MODEL = os.environ.get("REASONING_MODEL", "gpt-5-nano")
LOG_DIR = Path(os.environ.get("ORCHESTRATOR_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "topic_insights.jsonl"
MAX_STREAM_LENGTH = int(os.environ.get("TOPIC_STREAM_MAXLEN", "500"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY must be set for topic aggregator.")

client = OpenAI(api_key=OPENAI_API_KEY)


def setup_redis_client() -> redis.Redis:
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    client.ping()
    try:
        client.xgroup_create(name=INSIGHT_STREAM, groupname=GROUP, id="0", mkstream=True)
        print(f"[TopicAgg] Created consumer group '{GROUP}' on stream '{INSIGHT_STREAM}'.")
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    return client


def append_log(record: dict):
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class TopicBuffer:
    def __init__(self):
        self.entries: deque[dict] = deque()
        self.last_emit_ts: float = 0.0
        self.last_question: str = ""

    def add(self, entry: dict):
        self.entries.append(entry)
        now = time.time()
        while self.entries and (now - self.entries[0]["ts"]) > TIME_WINDOW:
            self.entries.popleft()

    def should_emit(self) -> bool:
        now = time.time()
        if len(self.entries) >= BUFFER_SIZE:
            return True
        if self.entries and (now - self.last_emit_ts) >= TIME_WINDOW and len(self.entries) >= 2:
            return True
        return False

    def snapshot(self) -> list[dict]:
        return list(self.entries)

    def mark_emitted(self, question: str):
        self.last_emit_ts = time.time()
        self.last_question = question
        if len(self.entries) > BUFFER_SIZE // 2:
            while len(self.entries) > BUFFER_SIZE // 2:
                self.entries.popleft()


buffers: dict[str, TopicBuffer] = {}


def summarize_topic(session_id: str, entries: list[dict]) -> dict | None:
    insights = []
    for item in entries:
        snippet = item.get("insights") or item.get("vlm_text")
        if snippet:
            insights.append(f"- {snippet.strip()}")
    if not insights:
        return None

    prompt = (
        "You are analyzing consecutive observations from a live video feed. Identify the dominant topic, "
        "summarize what is known, and craft ONE concise follow-up question to ask the person in the scene. "
        "If nothing actionable is present, return an empty question."
    )

    user_content = (
        f"Session: {session_id}\nObservations (most recent last):\n" + "\n".join(insights)
    )

    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "topic_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "summary": {"type": "string"},
                        "question": {"type": "string"},
                    },
                    "required": ["topic", "summary", "question"],
                    "additionalProperties": False,
                },
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not data.get("question"):
        return None
    return data


def process_event(rclient: redis.Redis, message_id: str, payload: dict):
    try:
        event = json.loads(payload.get("data", "{}"))
    except json.JSONDecodeError:
        rclient.xack(INSIGHT_STREAM, GROUP, message_id)
        return

    session_id = event.get("session_id")
    if not session_id:
        rclient.xack(INSIGHT_STREAM, GROUP, message_id)
        return

    buf = buffers.setdefault(session_id, TopicBuffer())
    buf.add({
        "ts": event.get("insight_ts", time.time()),
        "insights": event.get("insights", ""),
        "vlm_text": event.get("vlm_text", ""),
    })

    if buf.should_emit():
        now = time.time()
        if (now - buf.last_emit_ts) < COOLDOWN:
            rclient.xack(INSIGHT_STREAM, GROUP, message_id)
            return
        summary = summarize_topic(session_id, buf.snapshot())
        if summary and summary.get("question") != buf.last_question:
            record = {
                "session_id": session_id,
                "topic": summary.get("topic", "unknown"),
                "summary": summary.get("summary", ""),
                "question": summary.get("question", ""),
                "generated_ts": time.time(),
            }
            append_log(record)
            rclient.xadd(
                TOPIC_STREAM,
                {"data": json.dumps(record, ensure_ascii=False)},
                maxlen=MAX_STREAM_LENGTH,
                approximate=True,
            )
            buf.mark_emitted(record["question"])

    rclient.xack(INSIGHT_STREAM, GROUP, message_id)


def main():
    rclient = setup_redis_client()
    print(
        f"[TopicAgg] Listening on '{INSIGHT_STREAM}' as '{CONSUMER}', writing to '{TOPIC_STREAM}'"
    )

    running = True

    def _handle_signal(signum, *_):
        nonlocal running
        running = False
        print(f"[TopicAgg] Received signal {signum}. Shutting down...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while running:
        try:
            response = rclient.xreadgroup(
                GROUP,
                CONSUMER,
                {INSIGHT_STREAM: ">"},
                count=5,
                block=5000,
            )
            if not response:
                continue
            for _, messages in response:
                for message_id, payload in messages:
                    process_event(rclient, message_id, payload)
        except redis.exceptions.ConnectionError as exc:
            print(f"[TopicAgg] Redis connection lost: {exc}. Retrying in 5s...")
            time.sleep(5)
            rclient = setup_redis_client()
        except Exception as exc:
            print(f"[TopicAgg] Error: {exc}")
            time.sleep(1)

    print("[TopicAgg] Shutdown complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[TopicAgg] Interrupted.")
