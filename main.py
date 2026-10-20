import asyncio
import logging
import os
import re
import threading
import time

import aiohttp
import discord
import numpy as np
from discord.ext import commands, voice_recv
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
SERPAPI_KEY = os.environ["SERPAPI_KEY"]
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")  # small.en = more accurate, slower
SAFESEARCH = "off" if os.getenv("SAFESEARCH", "on").lower() == "off" else "active"

SERPAPI_URL = "https://serpapi.com/search.json"
TRIGGER = re.compile(r"\bshow\s+me\b[\s,]*(.+)", re.IGNORECASE)

SILENCE_SECS = 1.0             # this long without audio = end of a sentence
MAX_SECS = 15                  # force-flush very long speech
MIN_SECS = 0.5                 # ignore shorter blips
BYTES_PER_SEC = 48000 * 2 * 2  # Discord audio: 48 kHz, stereo, 16-bit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("krogbot")

whisper = WhisperModel(WHISPER_MODEL, compute_type="int8")

intents = discord.Intents.default()
intents.message_content = True  # also enable "Message Content Intent" in the Developer Portal
bot = commands.Bot(command_prefix="!", intents=intents)
sessions: dict[int, asyncio.Task] = {}


# ---------- image search ----------
async def image_loads(session: aiohttp.ClientSession, url: str) -> bool:
    """Quick check that a URL really serves an image (some sites block hotlinking)."""
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with session.get(url, timeout=timeout) as resp:
            return resp.status == 200 and resp.content_type.startswith("image/")
    except Exception:
        return False


async def search_image(query: str) -> str | None:
    """First Google Images result for the query, via SerpAPI."""
    params = {
        "engine": "google_images",
        "q": query,
        "safe": SAFESEARCH,
        "api_key": SERPAPI_KEY,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(SERPAPI_URL, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
        if data.get("error"):
            raise RuntimeError(data["error"])
        results = data.get("images_results", [])
        if not results:
            return None
        first = results[0]
        original = first.get("original")
        # Prefer the full-size image; fall back to Google's thumbnail if it won't load.
        if original and await image_loads(session, original):
            return original
        return first.get("thumbnail") or original


# ---------- audio capture ----------
class SpeechSink(voice_recv.AudioSink):
    """Collects audio per speaker; pop_ready() returns finished sentences."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.buffers: dict[int, tuple[bytearray, float]] = {}

    def wants_opus(self) -> bool:
        return False  # we want decoded PCM

    def write(self, user, data):  # called from a library thread
        if user is None:
            return
        with self.lock:
            buf, _ = self.buffers.get(user.id, (bytearray(), 0.0))
            buf.extend(data.pcm)
            self.buffers[user.id] = (buf, time.monotonic())

    def pop_ready(self) -> list[bytes]:
        ready, now = [], time.monotonic()
        with self.lock:
            for uid, (buf, last) in list(self.buffers.items()):
                if now - last >= SILENCE_SECS or len(buf) >= MAX_SECS * BYTES_PER_SEC:
                    if len(buf) > MIN_SECS * BYTES_PER_SEC:
                        ready.append(bytes(buf))
                    del self.buffers[uid]
        return ready

    def cleanup(self):
        with self.lock:
            self.buffers.clear()


# ---------- speech to text ----------
def to_whisper_audio(pcm: bytes) -> np.ndarray:
    """Discord PCM (48 kHz stereo int16) -> 16 kHz mono float32 for Whisper."""
    pcm = pcm[: len(pcm) // 4 * 4]
    mono = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2).mean(axis=1)
    mono = mono[: len(mono) // 3 * 3].reshape(-1, 3).mean(axis=1)
    return (mono / 32768.0).astype(np.float32)


def transcribe(pcm: bytes) -> str:
    segments, _ = whisper.transcribe(to_whisper_audio(pcm), language="en", vad_filter=True)
    return " ".join(s.text for s in segments).strip()


def parse_trigger(text: str) -> str | None:
    """Return what follows "show me", or None if the phrase wasn't said."""
    m = TRIGGER.search(text)
    if not m:
        return None
    return m.group(1).strip(" .,!?\"'") or None


# ---------- main loop ----------
async def listen_loop(vc, sink: SpeechSink, channel: discord.abc.Messageable):
    loop = asyncio.get_running_loop()
    while vc.is_connected():
        await asyncio.sleep(0.5)
        for pcm in sink.pop_ready():
            text = await loop.run_in_executor(None, transcribe, pcm)
            log.info("heard: %s", text)
            query = parse_trigger(text)
            if not query:
                continue
            try:
                url = await search_image(query)
            except Exception as e:
                log.exception("image search failed")
                await channel.send(f"Image search failed: {e}")
                continue
            if url:
                embed = discord.Embed(title=query)
                embed.set_image(url=url)
                await channel.send(embed=embed)
            else:
                await channel.send(f'Couldn\'t find an image for "{query}".')


@bot.command()
@commands.guild_only()
async def join(ctx: commands.Context):
    """Join your voice channel and listen for "show me ..."."""
    if not isinstance(ctx.author, discord.Member):
        return await ctx.send("Join a voice channel first.")

    if not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.send("Join a voice channel first.")

    if ctx.author.id in sessions:
        return await ctx.send("Already listening here. Use !leave first.")
    vc = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
    sink = SpeechSink()
    vc.listen(sink)
    task = asyncio.create_task(listen_loop(vc, sink, ctx.channel))
    task.add_done_callback(lambda _t, gid=ctx.author.id: sessions.pop(gid, None))
    sessions[ctx.author.id] = task
    await ctx.send(
        'Listening. Voice is transcribed to detect "show me ...", so make sure '
        "everyone in the channel is okay with that."
    )


@bot.command()
@commands.guild_only()
async def leave(ctx: commands.Context):
    """Stop listening and disconnect."""
    task = sessions.pop(ctx.author.id, None)
    if task:
        task.cancel()
    if ctx.voice_client:
        await ctx.voice_client.disconnect(force=False)

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
