# phoenix-realtime

Realtime voice chat with OpenAI's realtime API, in python, for Raspberry Pi Zero 2W with a Respeaker 2 Mic HAT.

Based on OpenAI's realtime API Python example: https://github.com/openai/openai-python/tree/main/examples/realtime

## Setup

1. Connect Respeaker 2 Mic HAT to Raspberry Pi Zero 2W.
2. Connect Raspberry Pi Zero 2W to power.
3. Install Raspian OS light bullseye.
4. Install git, pip, and python3.
5. Install respeaker driver (note this is for the version 1 HAT): https://github.com/respeaker/seeed-voicecard
6. Install this repo: `git clone https://github.com/SoulAuctioneer/phoenix-realtime.git`
7. Run install.sh
8. Add your OpenAI API key to local.env file
9. Run `./run.sh`