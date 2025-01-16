####################################################################
# Sample TUI app with a push to talk interface to the Realtime API #
####################################################################

from __future__ import annotations

import base64
import asyncio
import sys
from typing import Any, cast, TextIO
from typing_extensions import override

from textual import events
from audio_util import (
    CHANNELS, SAMPLE_RATE, INPUT_DEVICE_INDEX, AudioPlayerAsync,
    set_debug_callback
)
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, RichLog
from textual.reactive import reactive
from textual.containers import Container

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection


class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)

    @override
    def render(self) -> str:
        status = (
            "🔴 Recording... (Press K to stop)" if self.is_recording else "⚪ Press K to start recording (Q to quit)"
        )
        return status


class UIStdoutRedirector:
    """A file-like object that redirects stdout to the UI."""
    def __init__(self, app: RealtimeApp):
        self.app = app

    def write(self, text: str) -> int:
        if self.app.is_running:
            try:
                bottom_pane = self.app.query_one("#bottom-pane", RichLog)
                bottom_pane.write(text)
            except Exception:
                # Fallback to original stdout if UI isn't ready
                sys.__stdout__.write(text)
        else:
            # Fallback to original stdout if app isn't running
            sys.__stdout__.write(text)
        return len(text)

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        return sys.__stdout__.fileno()

    def isatty(self) -> bool:
        return sys.__stdout__.isatty()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
        }

        #bottom-pane {
            width: 100%;
            height: 82%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
    """

    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI()
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        
        # Set up debug callback
        def debug_to_ui(msg: str):
            if self.is_running:
                bottom_pane = self.query_one("#bottom-pane", RichLog)
                bottom_pane.write(msg)
        set_debug_callback(debug_to_ui)

        # Redirect stdout to UI
        sys.stdout = UIStdoutRedirector(self)

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview-2024-10-01",
            voice="alloy",
            speed=1.0
        ) as conn:
            self.connection = conn
            self.connected.set()
            bottom_pane = self.query_one("#bottom-pane", RichLog)
            bottom_pane.write("[INFO] Connected to Realtime API\n")

            # Configure session with both modalities and turn detection
            await conn.session.update(session={
                "modalities": ["text", "audio"],
                "turn_detection": {"type": "server_vad"}
            })
            bottom_pane.write("[INFO] Configured modalities and server VAD\n")

            acc_items: dict[str, Any] = {}

            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    session_display = self.query_one(SessionDisplay)
                    assert event.session.id is not None
                    session_display.session_id = event.session.id
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.write(f"[DEBUG] Session created: {event.session.id}\n")
                    bottom_pane.write("[INFO] Ready to record. Press K to start.\n")
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.write(f"[DEBUG] Session updated: {event.session}\n")
                    continue

                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id
                        bottom_pane = self.query_one("#bottom-pane", RichLog)
                        bottom_pane.write(f"[DEBUG] New audio response started: {event.item_id}\n")

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "response.audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.clear()
                    bottom_pane.write(f"[DEBUG] Transcript: {acc_items[event.item_id]}\n")
                    continue

                # Handle error events with more detail
                if event.type == "error":
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.write(f"[ERROR] {event.error.code}: {event.error.message}\n")
                    if hasattr(event.error, 'details'):
                        bottom_pane.write(f"[ERROR] Details: {event.error.details}\n")
                    continue

                # Debug any other events
                bottom_pane = self.query_one("#bottom-pane", RichLog)
                bottom_pane.write(f"[DEBUG] Other event: {event.type}\n")

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        import sounddevice as sd  # type: ignore

        sent_audio = False

        device_info = sd.query_devices()
        print(device_info)

        read_size = int(SAMPLE_RATE * 0.02)

        stream = sd.InputStream(
            device=INPUT_DEVICE_INDEX,  # Use ReSpeaker device for input
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
        )
        stream.start()

        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                await self.should_send_audio.wait()
                status_indicator.is_recording = True

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

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        bottom_pane = self.query_one("#bottom-pane", RichLog)
        
        if event.key == "q":
            bottom_pane.write("[INFO] Exiting application...\n")
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                self.should_send_audio.clear()
                status_indicator.is_recording = False
                bottom_pane.write("[INFO] Stopped recording\n")

                if self.session and self.session.turn_detection is None:
                    conn = await self._get_connection()
                    await conn.input_audio_buffer.commit()
                    bottom_pane.write("[INFO] Committed audio buffer\n")
                    await conn.response.create()
                    bottom_pane.write("[INFO] Requested response from model\n")
            else:
                # Only try to cancel if we have a connection
                if self.connection:
                    try:
                        await self.connection.send({"type": "response.cancel"})
                    except Exception as e:
                        # Silently handle cancellation failures
                        pass
                self.should_send_audio.set()
                status_indicator.is_recording = True
                bottom_pane.write("[INFO] Started recording\n")


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()
