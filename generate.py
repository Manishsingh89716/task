from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cuda")

def generate_audio(text, preset, output):
    inputs = processor(text, voice_preset=preset)
    for k, v in inputs.items():
        inputs[k] = v.to("cuda")
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy.squeeze()
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output, rate=sample_rate, data=audio_array)

generate_audio(text=" I have a silky smooth voice, and today I will tell you about the exercise regimen of the common sloth",
               preset="v2/hi_speaker_9",
               output="output.wav",
)