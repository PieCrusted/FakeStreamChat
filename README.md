# FakeStreamChat
Creating a basic fake stream chat for my friend group's DnD. The idea is to trascribe audio live, and use a combination of rule based systems and LLM generations to create fake twitch comments.

VERY IMPORTANT NOTE: This is built for intel mac, so yeah... hardware capabilities are on the weaker side.

The project uses OpenAI's Whisper base.en/tiny.en to transcribe audio into text. For text generation, uses RWKV RNN based text generator, as well as a rule based system to take keywords and generate fake text out of it.

Note: The project's basis is to run everything locally so we there shouldn't be any costs to money. But because of that, and I have a Intel MacBook Pro, it means the design/libary choices are based around the limited hardware.

#### TODO/Basic Plan/Parts/Flow Chart:

0. Set a basic UI control that can choose message/minute speeds, and rough macro themes like fighting, or chill talking npc modes
1. Record audio transcriptions and transcript audio into text
2. Send text into 3 different text generation systems
    1. Use a RNN RWKV model to give contextual responses based off the audio
    2. Use a rule based system to give spit out premade text if certain keywords are detected, like "chat", or "dnd",
       1. Uses preset theme to adjust the messages
    3. Use a rule based system to emotes spam, representing the chaotic nature of twitch chat messages
3. Create premade bot names, and output those messages by them
4. For now, it is ok to output to print, but later on figure out with creating a discord bot and live upload there.

#### Extras Plans/Changes:
Part 0. Mainly have the DM only be able to control some speed gauge or a concurrent viewers gauge, and base multiple functionality out of it

Part 0. If we did it by a speed gauge, we can also extrapolate the number of viewers with some +/- random percentage change. If we did it by number of viewers, we can also extrapolate some made up speed, and have that made up speed control the rest. So really in essence the DM would be able to control 1 gauge and if its the speed or the viewers gauge, it wouldn't matter too much.

Part 2. We can have an additional model similar to the rule based model in which sends preset messages, but have these messages be superchats to force the players to answer the questions/interact with the chat or steers the players into some directions.

Part 3. For the premade usernames, we could have the speed/viewers gauge input dicate how much of the pool it can grab usernames from. For example if there 1 chat per minute then it can grab from a pool of 5 usernames. This should be fairly easy to implemnet where we can take the list of names, randomize the order, and then select the first nth indexes as the pool of names and have the first nth grow as ths speed/viewer gauge increase, and lower n as it drops.


## Installation

Install the dependencies using the make file

```bash
make install
```
Prepare data in ```data/custom``` folder, in json format. Then run either:
```bash
make combine-training-data-only
```
```bash
make combine-training-data-test-split
```
```bash
make combine-training-data-test-split 0.3
```
The first command combines all the jsons in ```data/custom``` to ```data/train.json```

The second and third command combines and splits the jsons in ```data/custom``` to ```data/train.json``` and ```data/test.json```. It can take an specified argument between 0 to 1.0, where the argument is denoted as the percentage put into ```data/test.json```.

## Component Testing

To test this project run

```bash
make run-transcription
```
To test out the live transcription. So far it is only single threaded, so there are holes in the aduio where it is not recording because it is processing the audio. 

TODO: Use 2 threads, one to produce audio, and one to process audio, and have it communicate through a queue.

To test out segmented recording audio and processing transcriptions sepeartely to generate input data.

For recording audio into .wav files, run
```bash
make record-audio
```
Note that the audio recording will be in 1 minute clips, or whatever time you set it to, infinitely until doing Ctrl + C, in which it will continue to finish up recording the last minute it was still working on before stopping the program. I've found any audio losses from using very short clips, like 10 seconds long, to be negligible and unable to tell the difference by human ear.

To process the made .wav files into .txt transcriptions, run
```bash
make process-audio
```

If you are on Mac, and you wanted to record audio through the speaker, make sure you do ``make install-brew`` to install Blackhole-2ch and it will list out the full list of accessible so you could change the referenced devices in ``live_transcription.py`` and ``async_segmented_transcription.py`` to the devices of your own.

Additionally do **System Preferences > Sound > Output** and **System Preferences > Sound > Input** and select **Blackhole 2ch** for each. Just note that you won't be able to hear anything and your sound should be turned up to the max.

Then you could run 
```bash
make virtual-record-audio
```
to use the speakers to record audio into .wav files and still do the same 
```bash
make process-audio
```
to process the audio into transcriptions.
