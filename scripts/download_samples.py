import os
import urllib.request

# Directories
BASE_DIR = "test-audio-files"
DIRS = {
    "CREMA-D": os.path.join(BASE_DIR, "CREMA-D"),
    "CommonVoice": os.path.join(BASE_DIR, "CommonVoice"),
    "VoxCeleb": os.path.join(BASE_DIR, "VoxCeleb")
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# File URLs to download
DOWNLOADS = [
    # CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
    # Actor 1001 is a 51 year old male. Actor 1002 is a 21 year old female.
    {
        "url": "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/AudioWAV/1001_DFA_ANG_XX.wav",
        "path": os.path.join(DIRS["CREMA-D"], "1001_male_51.wav")
    },
    {
        "url": "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/AudioWAV/1002_DFA_ANG_XX.wav",
        "path": os.path.join(DIRS["CREMA-D"], "1002_female_21.wav")
    },
    {
        "url": "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/AudioWAV/1008_DFA_ANG_XX.wav",
        "path": os.path.join(DIRS["CREMA-D"], "1008_female_older.wav")
    },
    
    # Common Voice (Using Mozilla's DeepSpeech test audio repository as a proxy for CV samples)
    {
        "url": "https://raw.githubusercontent.com/mozilla/DeepSpeech/master/audio/2830-3980-0043.wav",
        "path": os.path.join(DIRS["CommonVoice"], "sample_1.wav")
    },
    {
        "url": "https://raw.githubusercontent.com/mozilla/DeepSpeech/master/audio/4507-16021-0012.wav",
        "path": os.path.join(DIRS["CommonVoice"], "sample_2.wav")
    },
    {
        "url": "https://raw.githubusercontent.com/mozilla/DeepSpeech/master/audio/8455-210777-0068.wav",
        "path": os.path.join(DIRS["CommonVoice"], "sample_3.wav")
    },

    # VoxCeleb (Using an open-source speech brain sample as proxy for VoxCeleb style conversational audio)
    {
        "url": "https://raw.githubusercontent.com/speechbrain/speechbrain/develop/samples/audio_samples/example1.wav",
        "path": os.path.join(DIRS["VoxCeleb"], "voxceleb_style_1.wav")
    },
    {
        "url": "https://raw.githubusercontent.com/speechbrain/speechbrain/develop/samples/audio_samples/example2.flac",
        "path": os.path.join(DIRS["VoxCeleb"], "voxceleb_style_2.flac")
    }
]

print("Downloading sample audio files for manual evaluation...")

for item in DOWNLOADS:
    print(f"Downloading {os.path.basename(item['path'])}...")
    try:
        urllib.request.urlretrieve(item["url"], item["path"])
        print(f"  -> Saved to {item['path']}")
    except Exception as e:
        print(f"  -> Failed to download: {e}")

print("\nDone! You can now test these files against the API.")
