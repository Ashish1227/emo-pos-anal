# from pyannote.audio import Pipeline
# import whisper
# import wave
# import contextlib

hf_token = "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"
# # Load Pyannote diarization pipeline
# diary_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",use_auth_token=hf_token)

# # Load Whisper model
# whisper_model = whisper.load_model("base")

# # Input audio file
# audio_file = "/Users/ashishgatreddi/Desktop/face-emo/Facial-Emotion-Recognition-using-OpenCV-and-Deepface/recorded_audio.wav"

# # Step 1: Perform Speaker Diarization
# diary_result = diary_pipeline(audio_file,num_speakers=2)

# # Get speaker segments
# speaker_segments = []
# for turn, _, speaker in diary_result.itertracks(yield_label=True):
#     speaker_segments.append((turn.start, turn.end, speaker))

# # Step 2: Perform Transcription
# with contextlib.closing(wave.open(audio_file, 'r')) as wf:
#     frame_rate = wf.getframerate()
#     n_frames = wf.getnframes()
#     duration = n_frames / float(frame_rate)

# # Transcribe entire audio
# transcription_result = whisper_model.transcribe(audio_file)
# transcription_segments = transcription_result['segments']

# # Step 3: Combine Diarization and Transcription
# output_text = []
# for start, end, speaker in speaker_segments:
#     speaker_text = ""
#     for segment in transcription_segments:
#         if start <= segment['start'] <= end:
#             speaker_text += segment['text'] + " "
#     output_text.append(f"Speaker {speaker}: \"{speaker_text.strip()}\"")

# # Step 4: Save the Output to a Text File
# with open("conversation_output.txt", "w") as f:
#     f.write("\n".join(output_text))

# print("Conversation diarization and transcription complete! Check 'conversation_output.txt'.")

from pyannote.audio import Pipeline
import whisper
import wave
import contextlib
from pydub import AudioSegment
from pydub.playback import play
import os

# Load Pyannote diarization pipeline
diary_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",use_auth_token=hf_token)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Input audio file
audio_file = "/Users/ashishgatreddi/Desktop/face-emo/Facial-Emotion-Recognition-using-OpenCV-and-Deepface/recorded_audio.wav"

# Step 1: Perform Speaker Diarization
diary_result = diary_pipeline(audio_file,num_speakers = 2)

# Get speaker segments
speaker_segments = []
for turn, _, speaker in diary_result.itertracks(yield_label=True):
    speaker_segments.append((turn.start, turn.end, speaker))

# Step 2: Perform Transcription
with contextlib.closing(wave.open(audio_file, 'r')) as wf:
    frame_rate = wf.getframerate()
    n_frames = wf.getnframes()
    duration = n_frames / float(frame_rate)

# Transcribe entire audio
transcription_result = whisper_model.transcribe(audio_file)
transcription_segments = transcription_result['segments']

# Step 3: Split Audio by Speaker and Transcription
output_text = []
audio = AudioSegment.from_wav(audio_file)
output_dir = "speaker_clips"
os.makedirs(output_dir, exist_ok=True)

for idx, (start, end, speaker) in enumerate(speaker_segments):
    speaker_text = ""
    for segment in transcription_segments:
        if start <= segment['start'] <= end:
            speaker_text += segment['text'] + " "

    # Save audio segment for each dialogue
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    speaker_audio = audio[start_ms:end_ms]
    clip_path = os.path.join(output_dir, f"{speaker}_{idx}.wav")
    speaker_audio.export(clip_path, format="wav")

    # Append text and placeholder for emotion detection
    output_text.append(f"Speaker {speaker}: \"{speaker_text.strip()}\" (audio emotion detected - TBD)")

# Step 4: Save the Output to a Text File
with open("conversation_output_with_emotions.txt", "w") as f:
    f.write("\n".join(output_text))

print("Conversation diarization, transcription, and audio splitting complete! Check 'conversation_output_with_emotions.txt' and 'speaker_clips' folder.")
