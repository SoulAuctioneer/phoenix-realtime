from __future__ import annotations

import io
import sys
import base64
import asyncio
import threading
import ctypes
from typing import Callable, Awaitable, Optional, Tuple, Any

import numpy as np
#from pydub import AudioSegment
import pipewire as pw

from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from config import (
    AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE, AUDIO_CHUNK_LENGTH_S,
    AUDIO_INPUT_SAMPLE_RATE, AUDIO_OUTPUT_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_WIDTH,
    AUDIO_LATENCY_MODE, AUDIO_BLOCKSIZE_MULTIPLIER
)

# Global debug callback
debug_callback: Callable[[str], None] | None = None

def set_debug_callback(callback: Callable[[str], None]):
    global debug_callback
    debug_callback = callback

def debug_print(msg: str):
    if debug_callback:
        debug_callback(f"[DEBUG] {msg}\n")
    else:
        # Only print to stdout if no callback is set
        print(msg)

def debug_audio_devices():
    """Print detailed information about PipeWire audio devices."""
    try:
        with pw.Context() as context:
            core = context.connect()
            registry = core.get_registry()
            
            print("\n=== Audio Device Information ===")
            print("All devices:")
            
            def registry_event(registry, id, type, version, props):
                if type == pw.NODE_TYPE:
                    name = props.get("node.name", "Unknown")
                    desc = props.get("node.description", "No description")
                    media_class = props.get("media.class", "Unknown")
                    print(f"\nDevice ID: {id}")
                    print(f"  Name: {name}")
                    print(f"  Description: {desc}")
                    print(f"  Media Class: {media_class}")
            
            registry.add_listener(registry_event)
            core.sync()
            
    except Exception as e:
        print(f"Error querying audio devices: {e}")
    print("\n=== End Audio Device Information ===\n")

# def audio_to_pcm16_base64(audio_bytes: bytes) -> bytes:
#     # load the audio file from the byte stream
#     audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
#     print(f"Loaded audio: {audio.frame_rate=} {audio.channels=} {audio.sample_width=} {audio.frame_width=}")
#     # resample to input sample rate mono pcm16
#     pcm_audio = audio.set_frame_rate(AUDIO_INPUT_SAMPLE_RATE).set_channels(AUDIO_CHANNELS).set_sample_width(2).raw_data
#     return pcm_audio

class AudioPlayerAsync:
    def __init__(self):
        self.sample_rate = AUDIO_OUTPUT_SAMPLE_RATE
        self.queue = []
        self.lock = threading.Lock()
        self.playing = False
        self._frame_count = 0
        self.paused = False
        
        # Initialize PipeWire
        self.pw = pw.Context()
        self.core = self.pw.connect()
        self.stream = None
        self._setup_stream()
        
    def _setup_stream(self):
        # Calculate buffer size
        self.buffer_size = int(AUDIO_CHUNK_LENGTH_S * self.sample_rate * AUDIO_BLOCKSIZE_MULTIPLIER)
        
        # Create stream properties
        props = {
            "media.type": "Audio",
            "audio.format": "S16LE",  # 16-bit signed little-endian
            "audio.rate": self.sample_rate,
            "audio.channels": AUDIO_CHANNELS,
        }
        
        # Create the stream
        self.stream = self.pw.Stream(
            name="Phoenix Audio Player",
            properties=props,
            mode="output",
            target=AUDIO_OUTPUT_DEVICE
        )
        
        # Set up the callback
        self.stream.connect_listener("process", self._process_callback)
        
    def _process_callback(self, data: Any) -> None:
        if not data:
            return
            
        with self.lock:
            if self.paused:
                data.fill(0)
                return
                
            buffer_data = np.empty(0, dtype=np.int16)
            
            while len(buffer_data) < self.buffer_size and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = self.buffer_size - len(buffer_data)
                buffer_data = np.concatenate((buffer_data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])
            
            self._frame_count += len(buffer_data)
            
            if len(buffer_data) < self.buffer_size:
                buffer_data = np.concatenate((buffer_data, np.zeros(self.buffer_size - len(buffer_data), dtype=np.int16)))
            
            try:
                if AUDIO_CHANNELS == 2:
                    if len(buffer_data.shape) == 1:
                        buffer_data = np.column_stack((buffer_data, buffer_data))
                    data[:] = buffer_data
                else:
                    data[:] = buffer_data.reshape(-1, AUDIO_CHANNELS)
            except Exception as e:
                debug_print(f'Error in audio callback: {e}')
                data.fill(0)

    def reset_frame_count(self):
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def is_queue_empty(self) -> bool:
        with self.lock:
            return len(self.queue) == 0

    def add_data(self, data: bytes):
        with self.lock:
            np_data = np.frombuffer(data, dtype=np.int16)
            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def start(self):
        if not self.playing:
            self.playing = True
            self.stream.activate()

    def pause(self):
        with self.lock:
            self.paused = True

    def resume(self):
        with self.lock:
            self.paused = False

    def stop(self):
        self.playing = False
        self.stream.deactivate()
        with self.lock:
            self.queue = []
            self.paused = False

    def terminate(self):
        if self.stream:
            self.stream.disconnect()
        if self.core:
            self.core.disconnect()
        if self.pw:
            self.pw.disconnect()

class AudioInputStream:
    def __init__(self, sample_rate: int = AUDIO_INPUT_SAMPLE_RATE, channels: int = AUDIO_CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer = []
        self.lock = threading.Lock()
        
        # Initialize PipeWire
        self.pw = pw.Context()
        self.core = self.pw.connect()
        self.stream = None
        self._setup_stream()
        
    def _setup_stream(self):
        props = {
            "media.type": "Audio",
            "audio.format": "S16LE",
            "audio.rate": self.sample_rate,
            "audio.channels": self.channels,
        }
        
        self.stream = self.pw.Stream(
            name="Phoenix Audio Input",
            properties=props,
            mode="input",
            target=AUDIO_INPUT_DEVICE
        )
        
        self.stream.connect_listener("process", self._process_callback)
        
    def _process_callback(self, data: Any) -> None:
        if not data:
            return
            
        with self.lock:
            self.buffer.append(np.array(data, dtype=np.int16))
            
    @property
    def read_available(self) -> int:
        with self.lock:
            return sum(len(chunk) for chunk in self.buffer)
            
    def read(self, frames: int) -> Tuple[np.ndarray, Any]:
        with self.lock:
            if not self.buffer:
                return np.zeros(frames, dtype=np.int16), None
                
            data = np.concatenate(self.buffer)
            if len(data) >= frames:
                result = data[:frames]
                remaining = data[frames:]
                self.buffer = [remaining] if len(remaining) > 0 else []
                return result, None
            else:
                self.buffer = []
                return np.pad(data, (0, frames - len(data))), None
                
    def start(self):
        self.stream.activate()
        
    def stop(self):
        self.stream.deactivate()
        
    def close(self):
        if self.stream:
            self.stream.disconnect()
        if self.core:
            self.core.disconnect()
        if self.pw:
            self.pw.disconnect()

async def send_audio_worker_pipewire(
    connection: AsyncRealtimeConnection,
    should_send: Callable[[], bool] | None = None,
    start_send: Callable[[], Awaitable[None]] | None = None,
) -> None:
    sent_audio = False
    read_size = int(AUDIO_INPUT_SAMPLE_RATE * 0.02)
    
    stream = AudioInputStream()
    stream.start()
    
    try:
        while True:
            if stream.read_available < read_size:
                await asyncio.sleep(0)
                continue
                
            data, _ = stream.read(read_size)
            
            if should_send() if should_send else True:
                if not sent_audio and start_send:
                    await start_send()
                debug_print("Sending audio buffer")
                await connection.send(
                    {"type": "input_audio_buffer.append", "audio": base64.b64encode(data.tobytes()).decode("utf-8")}
                )
                sent_audio = True
                
            elif sent_audio:
                debug_print("Done recording, triggering inference")
                await connection.send({"type": "input_audio_buffer.commit"})
                debug_print("Committed audio buffer")
                await connection.send({"type": "response.create", "response": {}})
                debug_print("Created response")
                sent_audio = False
                
            await asyncio.sleep(0)
            
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
