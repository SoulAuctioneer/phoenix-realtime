from __future__ import annotations

import io
import sys
import base64
import asyncio
import threading
from typing import Callable, Awaitable, Optional, Tuple, Any, cast

import numpy as np
# import pyaudio
import sounddevice as sd
# from pydub import AudioSegment

from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from config import (
    AUDIO_INPUT_BUFFER_SIZE_MS, AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE, AUDIO_OUTPUT_CHUNK_LENGTH_S,
    AUDIO_INPUT_SAMPLE_RATE, AUDIO_OUTPUT_SAMPLE_RATE, AUDIO_INPUT_CHANNELS, AUDIO_OUTPUT_CHANNELS, AUDIO_OUTPUT_WIDTH,
    AUDIO_OUTPUT_LATENCY_MODE, AUDIO_OUTPUT_BLOCKSIZE_MULTIPLIER
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

def debug_audio_devices(target_index: int | None = None):
    """Print detailed information about audio devices.
    
    Args:
        target_index: If provided, prints extra details about this specific device index
    """
    try:
        device_info = sd.query_devices()
        print("\n=== Audio Device Information ===")
        print("All devices:")
        for idx, device in enumerate(device_info):
            print(f"\nDevice {idx}: {device['name']}")
            print(f"  Max Input Channels: {device['max_input_channels']}")
            print(f"  Max Output Channels: {device['max_output_channels']}")
            print(f"  Default Sample Rate: {device['default_samplerate']}")
            
        if target_index is not None:
            try:
                target_info = sd.query_devices(target_index)
                print(f"\n=== Target Device (index {target_index}) Details ===")
                print(f"Name: {target_info['name']}")
                print(f"Max Input Channels: {target_info['max_input_channels']}")
                print(f"Max Output Channels: {target_info['max_output_channels']}")
                print(f"Default Sample Rate: {target_info['default_samplerate']}")
                print(f"Host API: {target_info['hostapi']}")
                print("Full device info:", target_info)
            except Exception as e:
                print(f"Error querying target device {target_index}: {e}")
                
    except Exception as e:
        print(f"Error querying audio devices: {e}")
    print("\n=== End Audio Device Information ===\n")


# NOTE: Unused, what's it for?
# def audio_to_pcm16_base64(audio_bytes: bytes) -> bytes:
#     # load the audio file from the byte stream
#     audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
#     print(f"Loaded audio: {audio.frame_rate=} {audio.channels=} {audio.sample_width=} {audio.frame_width=}")
#     # resample to input sample rate mono pcm16
#     pcm_audio = audio.set_frame_rate(AUDIO_INPUT_SAMPLE_RATE).set_channels(AUDIO_CHANNELS).set_sample_width(2).raw_data
#     return pcm_audio


class AudioPlayerAsync:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        
        # Calculate blocksize using the multiplier from config
        blocksize = int(AUDIO_OUTPUT_CHUNK_LENGTH_S * AUDIO_OUTPUT_SAMPLE_RATE * AUDIO_OUTPUT_BLOCKSIZE_MULTIPLIER)
        
        self.stream = sd.OutputStream(
            callback=self.callback,
            device=AUDIO_OUTPUT_DEVICE,
            samplerate=AUDIO_OUTPUT_SAMPLE_RATE,
            channels=AUDIO_OUTPUT_CHANNELS,
            dtype=np.int16,
            blocksize=blocksize,
            latency=AUDIO_OUTPUT_LATENCY_MODE  # Use latency mode from config
        )
        self.playing = False
        self._frame_count = 0
        self.paused = False  # Add paused state
        self.device = AUDIO_OUTPUT_DEVICE if AUDIO_OUTPUT_DEVICE is not None else sd.default.device[1]

    def callback(self, outdata, frames, time, status):
        if status:
            # Log any errors but continue playback
            debug_print(f'Audio callback status: {status}')
            
        with self.lock:
            if self.paused:  # If paused, output silence
                outdata.fill(0)
                return

            data = np.empty(0, dtype=np.int16)

            # get next item from queue if there is still space in the buffer
            while len(data) < frames and len(self.queue) > 0:
                item = self.queue.pop(0)
                frames_needed = frames - len(data)
                data = np.concatenate((data, item[:frames_needed]))
                if len(item) > frames_needed:
                    self.queue.insert(0, item[frames_needed:])

            self._frame_count += len(data)

            # fill the rest of the frames with zeros if there is no more data
            if len(data) < frames:
                data = np.concatenate((data, np.zeros(frames - len(data), dtype=np.int16)))

            try:
                # Ensure the output data has the correct shape for stereo
                if AUDIO_OUTPUT_CHANNELS == 2:
                    # Duplicate mono data to both channels if input is mono
                    if len(data.shape) == 1:
                        data = np.column_stack((data, data))
                    outdata[:] = data
                else:
                    outdata[:] = data.reshape(-1, AUDIO_OUTPUT_CHANNELS)
            except Exception as e:
                debug_print(f'Error in audio callback: {e}')
                outdata.fill(0)  # Output silence on error

    def reset_frame_count(self):
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def is_queue_empty(self) -> bool:
        """Check if there is any audio data left to play"""
        with self.lock:
            return len(self.queue) == 0

    def add_data(self, data: bytes):
        with self.lock:
            # bytes is pcm16 single channel audio data, convert to numpy array
            np_data = np.frombuffer(data, dtype=np.int16)
            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def start(self):
        self.playing = True
        self.stream.start()

    def pause(self):
        """Pause audio playback without clearing the queue"""
        with self.lock:
            self.paused = True

    def resume(self):
        """Resume audio playback"""
        with self.lock:
            self.paused = False

    def stop(self):
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = []
            self.paused = False  # Reset pause state when stopping

    def terminate(self):
        self.stream.close()

class AudioCaptureAsync:
    """Asynchronous audio capture class using sounddevice."""
    
    def __init__(self):
        self.stream = None
        self.is_recording = asyncio.Event()
        self.is_running = True
        self.device = AUDIO_INPUT_DEVICE if AUDIO_INPUT_DEVICE is not None else sd.default.device[0]
        self.capture_read_size = int(AUDIO_INPUT_SAMPLE_RATE * (AUDIO_INPUT_BUFFER_SIZE_MS / 1000))  # Convert ms to seconds

    def start(self) -> None:
        """Start recording audio."""
        if self.stream is not None:
            self.stop()

        self.stream = sd.InputStream(
            device=self.device,
            channels=AUDIO_INPUT_CHANNELS,
            samplerate=AUDIO_INPUT_SAMPLE_RATE,
            dtype="int16",
        )
        self.stream.start()
        self.is_recording.set()

    def stop(self) -> None:
        """Stop recording audio."""
        self.is_recording.clear()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def terminate(self) -> None:
        """Clean up resources."""
        self.is_running = False
        self.stop()

    async def read_chunk(self) -> tuple[np.ndarray, None]:
        """Read a chunk of audio data.
        
        Returns:
            Tuple of (audio_data, overflow) where audio_data is a numpy array of PCM data
        """
        if not self.is_recording.is_set() or self.stream is None:
            return np.array([], dtype=np.int16), None

        try:
            # Wait for sufficient audio to be available
            while self.stream is not None and self.stream.read_available < self.capture_read_size:
                await asyncio.sleep(0)
                if not self.is_recording.is_set() or self.stream is None:
                    return np.array([], dtype=np.int16), None

            # Double check stream is still valid
            if self.stream is None:
                return np.array([], dtype=np.int16), None

            data, overflow = self.stream.read(self.capture_read_size)
            # Ensure data is flattened to 1D array
            if len(data.shape) > 1:
                data = data.flatten()
            return data, overflow
        except Exception as e:
            debug_print(f"Error reading audio chunk: {e}")
            return np.array([], dtype=np.int16), None
