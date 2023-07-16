import sys
from src.utils import read_audio_file
from src.audio_processing import jump_height_from_audio

def main():
    if len(sys.argv) != 2:
        print("Usage: python run.py path/to/wav/file")
        sys.exit(1)

    file_path = sys.argv[1]

    audio, audio_fs = read_audio_file(file_path)
    jump_height = jump_height_from_audio(audio, audio_fs)
    if jump_height > 0:
        print(f'Estimated jump height: {jump_height:.2f} cm')

if __name__ == '__main__':
    main()