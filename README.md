# Speech-to-text and text-to-speech

Given a video, duplicate the audio in that video using an LLM voice. Do this by using `stt` to produce an `srt` file of the video, and then `tts` to create the audio.

I try and create meaningful chunks of text in the `srt` based on transitions in the video.

## Using

Start using this in the command line with:
source venv/bin/activate

Create a `.srt` from an MP4 like so:

python stt.py foo.mp4 foo.srt

Create a `.wav` file from the `.srt` file like so:

python tts.py foo.srt

## Past

I may have used some combination of these:

pip install kokoro soundfile pydub
pip install transformers accelerate datasets

## Attribution

Built using urgent prompts to an LLM.
