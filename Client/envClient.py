# Run this file after starting the server to begin training a model using the playSpace.Game class

import asyncio
from ws_client import ws_client
from playSpace import Game

# Step 1: define the handlers for your client.
# I.e. When an object is sent with a method name, create a function to be run when that message is received.

async def say(m):
    print(m['params']['text'])

# Advances the game by 1 time step and sends the current state data to server
async def step(m):
    done, reward, obs = game.step(m['params']['action'])
    await myNewClient.send_message("stateRewObs", {"reward": reward,
                                                   "obs": obs,
                                                   "done": done})
    if done:
        game.reset()

handlers = {
    "say": say,
    "step": step
}

# Step 2: create a new client.
# Include the value of the host, port, and the handlers object you just created.
# Beyond this, the client is running.
myNewClient = ws_client("localhost", 3000, handlers)
myNewClient.set_list_mode(True)
myNewClient.set_debug_mode(False)

game = Game()

# Step 3: write code to interact with a server.
async def acceptInput():
    test = input()
    return test

# Sends start command to server for training, then loops asyncio.sleep() to keep the event loop from finishing
async def send_messages():
    await myNewClient.send_message("start", {"init": [game.actions, game.minObsVal, game.maxObsVal, game.numObs, game.obsType]})
    while True:
        await asyncio.sleep(10)

async def main():

    await myNewClient.connect()
    handle_messages_task = asyncio.create_task(myNewClient.handle_messages())
    send_messages_task = asyncio.create_task(send_messages())
    # pygame.init()
    await asyncio.gather(handle_messages_task, send_messages_task)

if __name__ == "__main__":
    asyncio.run(main())
