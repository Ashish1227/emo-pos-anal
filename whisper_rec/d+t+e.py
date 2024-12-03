from pyannote.audio import Pipeline
import whisper
import wave
import contextlib
from pydub import AudioSegment
import os
from speechbrain.pretrained import EncoderClassifier
from huggingface_hub import login
from speechbrain.inference.interfaces import foreign_class

hf_token = "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"
login(token=hf_token)

# Load Pyannote diarization and Whisper model
diary_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",use_auth_token=hf_token)
whisper_model = whisper.load_model("base")

# Initialize the classifier
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
emotion_classifier = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", savedir="tmp")

# Input audio file
audio_file = "/Users/ashishgatreddi/Desktop/emo-pos-anal/fer_deepface/recorded_audio.wav"

def emotion(file_path):
   
    if file_path:
        # Classify the file
        out_prob, score, index, text_lab = classifier.classify_file(file_path)
        if isinstance(text_lab, list):
            text_lab = text_lab[0]
        # Map the original labels to the desired categories
        emotion_mapping = {
            'neu': 'Neutral',
            'ang': 'Angry',
            'hap': 'Happy',
            'sad': 'Sadness'
        }
        # Get the corresponding category from the mapping
        emotion_category = emotion_mapping.get(text_lab, 'Unknown')
        
        emotion_category = emotion_mapping.get(text_lab, 'Unknown')
        # Return the emotion category
        return emotion_category
    else:
        return "Please provide the path to an audio file."
    
# Step 1: Speaker Diarization (Limit to 1 speaker)
diary_result = diary_pipeline(audio_file, num_speakers=1)
speaker_segments = []
for turn, _, speaker in diary_result.itertracks(yield_label=True):
    speaker_segments.append((turn.start, turn.end, speaker))

# Step 2: Transcription
with contextlib.closing(wave.open(audio_file, 'r')) as wf:
    frame_rate = wf.getframerate()
    n_frames = wf.getnframes()
    duration = n_frames / float(frame_rate)

transcription_result = whisper_model.transcribe(audio_file)
transcription_segments = transcription_result['segments']

# Step 3: Split Audio and Perform Emotion Analysis
output_text = []
audio = AudioSegment.from_wav(audio_file)
output_dir = "speaker_clips_3"
os.makedirs(output_dir, exist_ok=True)

for idx, (start, end, speaker) in enumerate(speaker_segments):
    speaker_text = []
    for segment in transcription_segments:
        if start <= segment['start'] <= end:
            speaker_text.append({
                "text": segment['text'],
                "start_time": segment['start'],
                "end_time": segment['end']
            })

    # Save audio segment
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    speaker_audio = audio[start_ms:end_ms]
    clip_path = os.path.join(output_dir, f"{speaker}_{idx}.wav")
    speaker_audio.export(clip_path, format="wav")

    # Emotion detection
    em = emotion(clip_path)

    # Append text with emotion and timestamps
    for sentence in speaker_text:
        output_text.append(
            f"Speaker {speaker}: \"{sentence['text'].strip()}\" "
            f"(start: {sentence['start_time']:.2f}s, end: {sentence['end_time']:.2f}s, audio emotion detected - {em})"
        )

# Step 4: Save Output
with open("conversation_output_with_emotions2.txt", "w") as f:
    f.write("\n".join(output_text))

print("Diarization, transcription, and emotion analysis complete!")
