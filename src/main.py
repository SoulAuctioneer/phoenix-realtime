import asyncio
from push_to_talk_app import RealtimeApp

async def main():
    app = RealtimeApp()
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())
