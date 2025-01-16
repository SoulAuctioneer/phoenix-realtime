import asyncio
from audio_util import debug_audio_devices
from realtime_api_text_test import run_text_test
from push_to_talk_app import RealtimeApp

async def main():
    # First run audio device debug
    print("\nRunning audio device debug...")
    debug_audio_devices(target_index=1)  # Show details for ReSpeaker device
    
    # Then run text test
    print("\nRunning text test...")
    await run_text_test()
    
    # Finally run the push to talk app
    print("\nStarting push to talk app...")
    app = RealtimeApp()
    app.run()

if __name__ == "__main__":
    asyncio.run(main())
