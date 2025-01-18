####################################################################
# Command line push to talk interface to the Realtime API #
####################################################################

from __future__ import annotations

import base64
import asyncio
import sys
from typing import Any, cast

from audio_util import (
    INPUT_DEVICE_INDEX, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS,
    AudioPlayerAsync, set_debug_callback
)

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_VOICE, OPENAI_MODALITIES,
    OPENAI_INSTRUCTIONS, OPENAI_TRANSCRIPTION_MODEL, OPENAI_TURN_DETECTION,
    DISABLE_RECORDING_DURING_PLAYBACK
)

class RealtimeApp:
    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event
    is_playing_audio: asyncio.Event
    is_recording: bool
    running: bool
    audio_monitor_task: asyncio.Task | None

    @staticmethod
    def log(msg: str) -> None:
        """Write a message to stdout and flush."""
        print(msg, flush=True)

    @staticmethod
    def log_with_breaks(msg: str) -> None:
        """Write a message to stdout with line breaks before and after."""
        print(f"\n{msg}\n", flush=True)

    @staticmethod
    def write_transcript(msg: str) -> None:
        """Write transcript text without debug info."""
        print(msg, end="", flush=True)

    def log_event(self, event: Any) -> None:
        """Log event details based on event type."""
        event_type = event.type
        log_msg = f"[EVENT] {event_type}"

        if event_type == "error":
            log_msg += f": {event.error.message}"
        elif event_type == "response.audio_transcript.delta":
            return
            log_msg += f": {event.delta}"
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

        self.log(log_msg)

    def __init__(self) -> None:
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.is_playing_audio = asyncio.Event()
        self.is_recording = False
        self.running = True
        self.audio_monitor_task = None
        
        # Set up debug callback
        def debug_to_stdout(msg: str):
            self.log(msg)
        set_debug_callback(debug_to_stdout)

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(
            model=OPENAI_MODEL,
        ) as conn:
            self.connection = conn
            self.connected.set()
            self.log("[INFO] Connected to Realtime API")

            # Use turn detection setting from config
            await conn.session.update(session={
                "turn_detection": OPENAI_TURN_DETECTION,
                "modalities": OPENAI_MODALITIES,
                "voice": OPENAI_VOICE,
                "input_audio_transcription": {"model": OPENAI_TRANSCRIPTION_MODEL},
                "instructions": OPENAI_INSTRUCTIONS
            })
            self.log("[INFO] Updated session settings")

            acc_items: dict[str, Any] = {}

            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    self.log_event(event)
                    self.log("[INFO] Ready to record. Press K to start, Q to quit.")
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    self.log_event(event)
                    continue

                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id
                        self.log_event(event)
                        self.is_playing_audio.set()

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "input_audio_buffer.speech_started":
                    self.log_event(event)
                    if not DISABLE_RECORDING_DURING_PLAYBACK or not self.is_playing_audio.is_set():
                        self.log_with_breaks("[DEBUG] Speech detected, cancelling ongoing response and clearing audio buffer")
                        # Cancel ongoing response and clear audio buffer when speech starts
                        asyncio.create_task(self.connection.send({"type": "response.cancel"}))
                        self.audio_player.stop()
                    continue

                if event.type == "response.done":
                    self.log_event(event)
                    continue

                if event.type == "response.audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    self.log_event(event)
                    continue

                # Handle error events with more detail
                if event.type == "error":
                    self.log_event(event)
                    continue

                # Add detailed handling for conversation and response events
                if event.type == "conversation.item.created":
                    self.log_event(event)
                    continue

                if event.type == "conversation.item.input_audio_transcription.completed":
                    self.log_event(event)
                    continue

                if event.type == "response.created":
                    self.log_event(event)
                    continue

                if event.type == "response.output_item.added":
                    self.log_event(event)
                    continue

                if event.type == "response.content_part.added":
                    self.log_event(event)
                    continue

                if event.type == "response.audio.done":
                    self.log_event(event)
                    continue

                if event.type == "response.audio_transcript.done":
                    self.log_event(event)
                    continue

                if event.type == "response.content_part.done":
                    self.log_event(event)
                    continue

                if event.type == "response.output_item.done":
                    self.log_event(event)
                    continue

                if event.type == "input_audio_buffer.speech_started":
                    self.log_event(event)
                    continue

                if event.type == "input_audio_buffer.speech_stopped":
                    self.log_event(event)
                    continue

                if event.type == "input_audio_buffer.committed":
                    self.log_event(event)
                    continue

                if event.type == "rate_limits.updated":
                    self.log_event(event)
                    continue

                # Fallback for any unhandled events
                self.log_event(event)

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore

        sent_audio = False

        device_info = sd.query_devices()
        self.log(str(device_info))

        read_size = int(AUDIO_SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            device=INPUT_DEVICE_INDEX,  # Use ReSpeaker device for input
            channels=AUDIO_CHANNELS,
            samplerate=AUDIO_SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        try:
            while self.running:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                # Only check recording during playback if the feature is enabled
                if DISABLE_RECORDING_DURING_PLAYBACK and self.is_playing_audio.is_set():
                    self.log_with_breaks(f"[DEBUG] Recording during playback is disabled")
                    if self.is_recording:
                        self.is_recording = False
                        self.log_with_breaks("[INFO] Paused recording while audio is playing")
                    sent_audio = False
                    await asyncio.sleep(0)
                    continue

                # Wait for recording to be enabled
                if not self.should_send_audio.is_set():
                    if self.is_recording:
                        self.is_recording = False
                    sent_audio = False
                    await asyncio.sleep(0)
                    continue

                data, _ = stream.read(read_size)
                self.is_recording = True

                connection = await self._get_connection()
                if not sent_audio:
                    sent_audio = True

                await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))
                await asyncio.sleep(0)

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()

    async def handle_input(self) -> None:
        """Handle keyboard input in a separate task"""
        self.log("[INFO] Ready for input. Press K to start/stop recording, Q to quit")
        
        while self.running:
            try:
                # Read a single character
                key = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.read, 1)
                
                if key.lower() == 'q':
                    self.log_with_breaks("[INFO] Exiting application...")
                    self.running = False
                    self.should_send_audio.clear()
                    if self.connection:
                        try:
                            await self.connection.send({"type": "response.cancel"})
                            await self.connection.close()
                        except:
                            pass
                    # Stop audio playback
                    self.audio_player.stop()
                    return

                elif key.lower() == 'k':
                    if self.is_recording:
                        self.should_send_audio.clear()
                        self.is_recording = False
                        self.log_with_breaks("[INFO] Stopped recording")

                        if self.session and self.session.turn_detection is None:
                            conn = await self._get_connection()
                            await conn.input_audio_buffer.commit()
                            self.log("[INFO] Committed audio buffer")
                            await conn.response.create()
                            self.log("[INFO] Requested response from model")
                    else:
                        # Only try to cancel if we have a connection
                        if self.connection:
                            try:
                                await self.connection.send({"type": "response.cancel"})
                            except Exception as e:
                                # Silently handle cancellation failures
                                pass
                        
                        self.should_send_audio.set()
                        self.is_recording = True
                        self.log_with_breaks("[INFO] Started recording")

            except Exception as e:
                self.log_with_breaks(f"[ERROR] Input handling error: {e}")
                await asyncio.sleep(0.1)  # Prevent tight loop on error

    async def monitor_audio_playback(self) -> None:
        """Monitor audio playback and update is_playing_audio flag"""
        while self.running:
            if self.is_playing_audio.is_set() and self.audio_player.is_queue_empty():
                # Give a small buffer to ensure audio is fully played
                await asyncio.sleep(0.1)
                if self.audio_player.is_queue_empty():
                    self.is_playing_audio.clear()
                    self.log("[DEBUG] Audio playback complete")
            await asyncio.sleep(0.05)

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
            self.log(f"[ERROR] Application error: {e}")
        finally:
            self.running = False
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
        app.log_with_breaks("[INFO] Interrupted by user")
    except Exception as e:
        app.log(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
