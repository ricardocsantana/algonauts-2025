
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Multilingual examples
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

pt_text = "Olá, como estás? Isto é uma experiência de síntese de voz com uma nova ferramenta."
wav_pt = multilingual_model.generate(pt_text, language_id="pt")
ta.save("test-pt.wav", wav_pt, multilingual_model.sr)