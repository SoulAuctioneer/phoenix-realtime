####################################################################
# Command line push to talk interface to the Realtime API #
####################################################################

from __future__ import annotations

import base64
import asyncio
import sys
from typing import Any, cast

from audio_util import (
    INPUT_DEVICE_INDEX, AUDIO_INPUT_SAMPLE_RATE, AUDIO_CHANNELS,
    AudioPlayerAsync, set_debug_callback
)

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_VOICE, OPENAI_MODALITIES,
    OPENAI_INSTRUCTIONS, OPENAI_TRANSCRIPTION_MODEL, OPENAI_TURN_DETECTION,
    ALLOW_RECORDING_DURING_PLAYBACK
)

# Maximum number of connection retries
MAX_RETRIES = 3
# Delay between retries (in seconds)
RETRY_DELAY = 2

class RealtimeApp:
    client: AsyncOpenAI
    is_recording: asyncio.Event
    is_connected: asyncio.Event
    is_playing_audio: asyncio.Event
    is_response_active: asyncio.Event  # Track if there's an active response
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    is_running: bool
    audio_monitor_task: asyncio.Task | None

    def log(self, msg: str) -> None:
        """Write a message to stdout and flush."""
        print(f"[INFO] {msg}", flush=True)

    def log_error(self, msg: str) -> None:
        """Write an error message to stdout and flush."""
        print(f"[ERROR] {msg}", flush=True)

    def log_event(self, event: Any) -> None:
        """Log event details based on event type."""
        event_type = event.type
        log_msg = ""
        if event_type == "error":
            self.log_error(f"[EVENT]: {event.error.message}")
            return
        elif event_type == "response.audio_transcript.delta":
            return
            # log_msg += f": {event.delta}"
        elif event_type == "response.audio.delta":
            return
            # log_msg += f": {event.delta}"
        elif event_type == "response.audio_transcript.done":
            log_msg += f": {event.transcript}"
        elif event_type == "conversation.item.input_audio_transcription.completed":
            log_msg += f": {event.transcript}"
        elif event_type == "input_audio_buffer.speech_started":
            log_msg += f" at {event.audio_start_ms}ms"
        elif event_type == "input_audio_buffer.speech_stopped":
            log_msg += f" at {event.audio_end_ms}ms"
        elif event_type == "rate_limits.updated":
            limits = [f"{limit.name}={limit.remaining}/{limit.limit}" for limit in event.rate_limits]
            log_msg += f": {', '.join(limits)}"
        elif event_type == "session.created" or event_type == "session.updated":
            log_msg += f": {event.session}"
        elif event_type == "response.created":
            log_msg += f": {event.response.id}"
        elif event_type == "response.done":
            log_msg += f": {event.response.id}"
            if event.response.usage:
                log_msg += f" (total tokens: {event.response.usage.total_tokens})"

        print(f"[EVENT] {event_type} {log_msg}", flush=True)

    def __init__(self) -> None:
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.is_recording = asyncio.Event()
        self.is_connected = asyncio.Event()
        self.is_playing_audio = asyncio.Event()
        self.is_response_active = asyncio.Event()
        self.is_running = True
        self.audio_monitor_task = None
        
        # Set up debug callback
        def debug_to_stdout(msg: str):
            self.log(msg)
        set_debug_callback(debug_to_stdout)

    async def handle_realtime_connection(self) -> None:
        retry_count = 0
        while retry_count < MAX_RETRIES and self.is_running:
            try:
                async with self.client.beta.realtime.connect(
                    model=OPENAI_MODEL,
                ) as conn:
                    self.connection = conn
                    self.is_connected.set()
                    self.log("Connected to Realtime API")
                    retry_count = 0  # Reset retry count on successful connection

                    # Use turn detection setting from config
                    await conn.session.update(session={
                        "turn_detection": OPENAI_TURN_DETECTION,
                        "modalities": OPENAI_MODALITIES,
                        "voice": OPENAI_VOICE,
                        "input_audio_transcription": {"model": OPENAI_TRANSCRIPTION_MODEL},
                        "instructions": OPENAI_INSTRUCTIONS
                    })
                    self.log("Updated session settings")

                    await self._process_events()

            except Exception as e:
                if not self.is_running:
                    return
                
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    self.log_error(f"Connection failed: {e}. Retrying in {RETRY_DELAY} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                    self.is_connected.clear()
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    self.log_error(f"Failed to establish connection after {MAX_RETRIES} attempts: {e}")
                    self.is_running = False
                    return

    async def _handle_session_event(self, event: Any) -> None:
        """Handle session.created and session.updated events."""
        self.session = event.session
        self.log("Ready to record. Press K to start, Q to quit.")

    async def _handle_audio_delta(self, event: Any) -> None:
        """Handle response.audio.delta events."""
        if event.item_id != self.last_audio_item_id:
            self.audio_player.reset_frame_count()
            self.last_audio_item_id = event.item_id
            self.is_playing_audio.set()

        bytes_data = base64.b64decode(event.delta)
        self.audio_player.add_data(bytes_data)

    async def _handle_speech_started(self, event: Any) -> None:
        """Handle input_audio_buffer.speech_started events."""
        if not ALLOW_RECORDING_DURING_PLAYBACK:
            return

        if self.is_response_active.is_set() or self.is_playing_audio.is_set():
            self.log("Speech detected during active response, cancelling response")
            if self.is_response_active.is_set():
                asyncio.create_task(self.connection.send({"type": "response.cancel"}))
            if self.is_playing_audio.is_set():
                self.audio_player.stop()

    async def _process_events(self) -> None:
        """Process events from the realtime connection."""
        acc_items: dict[str, str] = {}

        try:
            async for event in self.connection:
                if not self.is_running:
                    break

                self.log_event(event)

                if event.type == "session.created" or event.type == "session.updated":
                    await self._handle_session_event(event)                
                elif event.type == "response.audio.delta":
                    await self._handle_audio_delta(event)
                elif event.type == "input_audio_buffer.speech_started":
                    await self._handle_speech_started(event)  
                elif event.type == "response.created":
                    self.is_response_active.set()
                elif event.type == "response.done":
                    self.is_response_active.clear()
                elif event.type == "response.audio_transcript.delta":
                    acc_items[event.item_id] = acc_items.get(event.item_id, "") + event.delta

        except Exception as e:
            if not self.is_running:
                return
            self.log_error(f"Connection error: {e}")
            raise  # Re-raise to trigger retry

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.is_connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore
        device_info = sd.query_devices()
        self.log(str(device_info))
        read_size = int(AUDIO_INPUT_SAMPLE_RATE * 0.02)
        stream = sd.InputStream(
            device=INPUT_DEVICE_INDEX,
            channels=AUDIO_CHANNELS,
            samplerate=AUDIO_INPUT_SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        try:
            while self.is_running:
                # Wait for sufficient audio to be available
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                # If we're playing audio and we're not allowed to record during playback, skip
                if self.is_playing_audio.is_set() and not ALLOW_RECORDING_DURING_PLAYBACK:
                    await asyncio.sleep(0)
                    continue
                
                # Wait for recording to be enabled
                await self.is_recording.wait()

                # Read audio data
                data, _ = stream.read(read_size)
                connection = await self._get_connection()

                await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))
                await asyncio.sleep(0)

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

    async def monitor_audio_playback(self) -> None:
        """Monitor audio playback and update is_playing_audio flag"""
        while self.is_running:
            if self.is_playing_audio.is_set() and self.audio_player.is_queue_empty():
                # Give a small buffer to ensure audio is fully played
                await asyncio.sleep(0.1)
                if self.audio_player.is_queue_empty():
                    self.is_playing_audio.clear()
                    self.log("Audio playback complete")
            await asyncio.sleep(0.05)

    async def handle_input(self) -> None:
        """Handle keyboard input in a separate task"""
        self.log("Ready for input. Press K to start/stop recording, Q to quit")
        
        while self.is_running:
            try:
                # Read a single character
                # TODO: use keyboard library to handle key presses
                key = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.read, 1)                
                if key.lower() == 'q':
                    await self.exit()
                elif key.lower() == 'k':
                    await self.toggle_recording()

            except Exception as e:
                self.log_error(f"Input handling error: {e}")
                await asyncio.sleep(0.1)  # Prevent tight loop on error

    async def toggle_recording(self) -> None:
        if self.is_recording.is_set():
            await self.stop_recording()
        else:
            await self.start_recording()

    async def start_recording(self) -> None:
        # Cancel any ongoing response if we have a connection and an ongoing response
        if self.connection and self.is_response_active.is_set():
            try:
                await self.connection.send({"type": "response.cancel"})
            except Exception as e:
                pass
        self.is_recording.set()
        self.log("Started recording")

    async def stop_recording(self) -> None:
        self.is_recording.clear()
        self.log("Stopped recording")

        if self.session and self.session.turn_detection is None:
            conn = await self._get_connection()
            await conn.input_audio_buffer.commit()
            self.log("Committed audio buffer")
            await conn.response.create()
            self.log("Requested response from model")

    async def exit(self) -> None:
        self.log("Exiting application...")
        self.is_running = False
        self.is_recording.clear()
        if self.connection:
            try:
                await self.connection.send({"type": "response.cancel"})
                await self.connection.close()
            except:
                pass
        # Stop audio playback
        self.audio_player.stop()
        return

    async def run(self) -> None:
        """Run the application"""
        try:
            # Start the main tasks
            connection_task = asyncio.create_task(self.handle_realtime_connection())
            audio_task = asyncio.create_task(self.send_mic_audio())
            input_task = asyncio.create_task(self.handle_input())
            self.audio_monitor_task = asyncio.create_task(self.monitor_audio_playback())
            
            # Wait for any task to complete or an error
            done, pending = await asyncio.wait(
                [connection_task, audio_task, input_task, self.audio_monitor_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel all pending tasks when one completes
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            self.log_error(f"Application error: {e}")
        finally:
            self.is_running = False
            self.audio_player.terminate()
            # Ensure we stop the audio stream
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()

def main():
    """Main entry point"""
    app = RealtimeApp()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        app.log("Interrupted by user")
    except Exception as e:
        app.log(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
