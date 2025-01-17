import asyncio
from audio_util import debug_audio_devices, validate_audio_config, get_device_info
from realtime_api_text_test import run_text_test
from push_to_talk_app import RealtimeApp
import os

async def run_initial_tests():
    # First show current audio configuration
    print("\nDetected audio configuration:")
    print(get_device_info())
    
    # Then validate audio configuration
    print("\nValidating audio configuration...")
    if not validate_audio_config():
        print("ERROR: Invalid audio configuration. Please check your settings in local.env")
        print("You can copy local.env.template to local.env and adjust the settings.")
        print("\nAvailable devices:")
        debug_audio_devices()  # Show all devices to help user configure
        exit(1)
    
    # Then run audio device debug if requested
    if os.getenv('DEBUG_AUDIO', '').lower() in ('1', 'true', 'yes'):
        print("\nRunning audio device debug...")
        debug_audio_devices()
    
    # Then run text test
    # print("\nRunning text test...")
    # await run_text_test()

def main():
    # Optionally Run the async tests first
    asyncio.run(run_initial_tests())
    
    # Then run the push to talk app
    print("\nStarting push to talk app...")
    app = RealtimeApp()
    asyncio.run(app.start())

if __name__ == "__main__":
    main()
