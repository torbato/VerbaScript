import whisper
import os

whisper_package_path = os.path.dirname(whisper.__file__)
assets_path = os.path.join(whisper_package_path, "assets")
print(f"Whisper assets path is: {assets_path}")
print(f"Check for mel_filters.npz: {os.path.join(assets_path, 'mel_filters.npz')}")
print(f"Check for multilingual.tiktoken: {os.path.join(assets_path, 'multilingual.tiktoken')}")