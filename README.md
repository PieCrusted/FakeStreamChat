# FakeStreamChat
Creating a basic fake stream chat for my friend group's DnD. The idea is to trascribe audio live, and use a combination of rule based systems and LLM generations to create fake twitch comments.

The project uses OpenAI's Whisper base.en/tiny.en to transcribe audio into text. For text generation, uses RWKV RNN based text generator, as well as a rule based system to take keywords and generate fake text out of it.

Note: The project's basis is to run everything locally so we there shouldn't be any costs to money. But because of that, and I have a Intel MacBook Pro, it means the design/libary choices are based around the limited hardware.

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
Where the first combines all the jsons in ```data/custom``` to ```data/train.json```

The second and third combines and splits the jsons in ```data/custom``` to ```data/train.json``` and ```data/test.json```. It can take an specified argument between 0 to 1.0, where the argument is denoted as the percentage put into ```data/test.json```.

## Component Testing

To test this project run

```bash
  make run-transcription
```
To test out the live transcription. So far it is only single threaded, so there are holes in the aduio where it is not recording because it is processing the audio. 

TODO: Use 2 threads, one to produce audio, and one to process audio, and have it communicate through a queue.

