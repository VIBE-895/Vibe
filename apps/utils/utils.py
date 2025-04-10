# Created by guxu at 9/12/24
import os.path
import uuid

import ffmpeg


def extract_audio(input_video_path, output_audio_path):
    # Load the input video file
    stream = ffmpeg.input(input_video_path)

    # Extract audio and save to the output file
    audio = stream.audio
    output = ffmpeg.output(audio, output_audio_path)

    # Run the command
    ffmpeg.run(output)

def write_output(path, content):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    try:
        with open(path, 'w') as f:
            f.write(content)
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

def is_pdf(file_path):
    return file_path.lower().endswith('.pdf')

def is_image(file_path):
    return file_path.lower().endswith('.jpg') or file_path.lower().endswith('.png') or file_path.lower().endswith('.jpeg')

def gen_id():
    return str(uuid.uuid4())

if __name__ == '__main__':
    input_path = "../data/video/demo.webm"
    output_path = "../../data/audio/demo.mp3"

    # extract_audio(input_path, output_path)
    write_output(input_path)