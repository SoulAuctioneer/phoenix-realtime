import asyncio
from audio_util import debug_audio_devices, validate_audio_config, get_device_info
from realtime_api_text_test import run_text_test
from push_to_talk_app import RealtimeApp
from config import DEBUG_AUDIO

async def run_initial_tests():
    # First show current audio configuration
    print("\nDetected audio configuration:")
    print(get_device_info())
    
    # Then validate audio configuration
    print("\nValidating audio configuration...")
    if not validate_audio_config():
        print("ERROR: Invalid audio configuration. Please check your settings in config.py")
        print("\nAvailable devices:")
        debug_audio_devices()  # Show all devices to help user configure
        exit(1)
    
    # Then run audio device debug if requested
    if DEBUG_AUDIO:
        print("\nRunning audio device debug...")
        debug_audio_devices()

async def main():
    # Run the async tests first
    # await run_initial_tests()
    # Don't
    
    # Then run the push to talk app
    print("\nStarting push to talk app...")
    app = RealtimeApp()
    await app.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
