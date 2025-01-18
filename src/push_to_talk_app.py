####################################################################
# Command line push to talk interface to the Realtime API #
####################################################################

from __future__ import annotations

import base64
import asyncio
import sys
import signal
import termios
import tty
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
    OPENAI_INSTRUCTIONS, OPENAI_TRANSCRIPTION_MODEL, OPENAI_TURN_DETECTION
)

class RawTerminal:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, type, value, traceback):
        if self.old_settings:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

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
        
        # Set up debug callback
        def debug_to_stdout(msg: str):
            print(msg + '\n', end='')
        set_debug_callback(debug_to_stdout)

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(
            model=OPENAI_MODEL,
        ) as conn:
            self.connection = conn
            self.connected.set()
            print("[INFO] Connected to Realtime API\n")

            # Use turn detection setting from config
            await conn.session.update(session={
                "turn_detection": {"type": OPENAI_TURN_DETECTION} if OPENAI_TURN_DETECTION else None,
                "modalities": OPENAI_MODALITIES,
                "voice": OPENAI_VOICE,
                "input_audio_transcription": {"model": OPENAI_TRANSCRIPTION_MODEL},
                "instructions": OPENAI_INSTRUCTIONS
            })
            print("[INFO] Updated session settings\n")

            acc_items: dict[str, Any] = {}

            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    assert event.session.id is not None
                    print(f"[DEBUG] Session created: {event.session.id}\n")
                    print("[INFO] Ready to record. Press K to start, Q to quit.\n")
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    print(f"[DEBUG] Session updated: {event.session}\n")
                    continue

                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id
                        print(f"[DEBUG] New audio response started: {event.item_id}\n")
                        self.is_playing_audio.set()

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "response.done":
                    # Signal that audio playback is complete
                    self.is_playing_audio.clear()
                    print("[DEBUG] Response complete, resuming recording\n")
                    continue

                if event.type == "response.audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    print(event.delta)
                    continue

                # Handle error events with more detail
                if event.type == "error":
                    print(f"[ERROR] {event.error.code}: {event.error.message}\n")
                    if hasattr(event.error, 'details'):
                        print(f"[ERROR] Details: {event.error.details}\n")
                    continue

                # Add detailed handling for conversation and response events
                if event.type == "conversation.item.created":
                    print(f"[DEBUG] Conversation item created - ID: {event.item.id}, Role: {event.item.role}\n")
                    continue

                if event.type == "conversation.item.input_audio_transcription.completed":
                    if hasattr(event, 'transcript'):
                        print(f"Input Transcription Completed: {event.transcript}\n")
                    continue

                if event.type == "response.created":
                    print("[DEBUG] Response created\n")
                    continue

                if event.type == "response.output_item.added":
                    print(f"[DEBUG] Output item added - ID: {event.item.id}, Type: {event.item.type}\n")
                    if hasattr(event.item, 'content'):
                        print(f"[DEBUG] Content: {event.item.content}\n")
                    continue

                if event.type == "response.content_part.added":
                    print(f"[DEBUG] Content part added - Item ID: {event.item_id}\n")
                    if hasattr(event, 'content'):
                        print(f"[DEBUG] Content: {event.content}\n")
                    continue

                if event.type == "response.audio.done":
                    print("[DEBUG] Audio response complete\n")
                    continue

                if event.type == "response.audio_transcript.done":
                    print("[DEBUG] Audio transcript complete\n")
                    continue

                if event.type == "response.content_part.done":
                    print("[DEBUG] Content part complete\n")
                    continue

                if event.type == "response.output_item.done":
                    print("[DEBUG] Output item complete\n")
                    continue

                if event.type == "input_audio_buffer.speech_started":
                    print("[DEBUG] Speech started in audio buffer\n")
                    continue

                if event.type == "input_audio_buffer.speech_stopped":
                    print("[DEBUG] Speech stopped in audio buffer\n")
                    continue

                if event.type == "input_audio_buffer.committed":
                    print("[DEBUG] Audio buffer committed\n")
                    continue

                if event.type == "rate_limits.updated":
                    print(f"[DEBUG] Rate limits updated: {event.rate_limits}\n")
                    continue

                # Fallback for any unhandled events
                print(f"[DEBUG] Other event type: {event.type}\n")

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore

        sent_audio = False

        device_info = sd.query_devices()
        print(device_info)

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

                # Only process audio input if we should be sending AND we're not playing audio
                if not self.should_send_audio.is_set() or self.is_playing_audio.is_set():
                    sent_audio = False  # Reset sent_audio when we stop sending
                    await asyncio.sleep(0)
                    continue

                self.is_recording = True
                data, _ = stream.read(read_size)

                connection = await self._get_connection()
                if not sent_audio:
                    asyncio.create_task(connection.send({"type": "response.cancel"}))
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
        print("[INFO] Ready for input. Press K to start/stop recording, Q to quit\n")
        
        with RawTerminal():
            while self.running:
                try:
                    # Read a single character without requiring Enter
                    key = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.read, 1)
                    
                    if key.lower() == 'q':
                        print("\n[INFO] Exiting application...\n")
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
                        return  # Exit the input handling loop immediately

                    elif key.lower() == 'k':
                        if self.is_recording:
                            self.should_send_audio.clear()
                            self.is_recording = False
                            print("\n[INFO] Stopped recording\n")

                            if self.session and self.session.turn_detection is None:
                                conn = await self._get_connection()
                                await conn.input_audio_buffer.commit()
                                print("[INFO] Committed audio buffer\n")
                                await conn.response.create()
                                print("[INFO] Requested response from model\n")
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
                            print("\n[INFO] Started recording\n")

                except Exception as e:
                    print(f"\n[ERROR] Input handling error: {e}\n")
                    await asyncio.sleep(0.1)  # Prevent tight loop on error

    async def run(self) -> None:
        """Run the application"""
        try:
            # Start the main tasks
            connection_task = asyncio.create_task(self.handle_realtime_connection())
            audio_task = asyncio.create_task(self.send_mic_audio())
            input_task = asyncio.create_task(self.handle_input())
            
            # Wait for any task to complete or an error
            done, pending = await asyncio.wait(
                [connection_task, audio_task, input_task],
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
            print(f"[ERROR] Application error: {e}")
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
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
