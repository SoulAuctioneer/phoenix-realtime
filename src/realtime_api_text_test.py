import asyncio
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

async def run_text_test():
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async with client.beta.realtime.connect(model=OPENAI_MODEL) as connection:
        await connection.session.update(session={'modalities': ['text']})

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello! Then tell me a joke."}],
            }
        )
        await connection.response.create()

        async for event in connection:
            if event.type == 'response.text.delta':
                print(event.delta, flush=True, end="")

            elif event.type == 'response.text.done':
                print()

            elif event.type == "response.done":
                break
