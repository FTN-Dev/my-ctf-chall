#!/usr/bin/env python3
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import librosa
import soundfile as sf
from pydub import AudioSegment
import os

FLAG = "LYCORIS{1n1_4d4L4h_B4s1C_4Ud10}"
INPUT_MP3 = "chall.mp3"
OUTPUT_WAV = "chall.wav"
START_TIME = 5.0
DURATION = 30.0
SR = 22050

# parameter utama untuk memperjelas tulisan
GAIN_DB = 0.0          # jangan terlalu kecil, biar stego lebih jelas
FLAG_DURATION = 15.0   # durasi khusus teks
FONT_SIZE = 80         # font lebih besar biar tebal
MAX_AMP = 12.0         # amplitudo lebih tinggi biar kontras

def text_to_image(text, width, height, font_size=48):
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # kalau teks lebih panjang dari width → scale otomatis
    if w > width:
        scale = width / (w + 1e-9)
        font_size = int(font_size * scale)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.text(((width - w)//2, (height - h)//2), text, fill=255, font=font)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(img)

def image_to_mag(img, n_fft=2048, hop_length=512, max_amp=8.0):
    freq_bins = n_fft // 2 + 1
    pil = Image.fromarray(img)
    pil = pil.resize((img.shape[1], freq_bins))
    arr = np.array(pil).astype(np.float32) / 255.0
    return arr * max_amp

def mag_to_audio(mag, n_fft=2048, hop_length=512, n_iter=80):
    return librosa.griffinlim(mag, n_iter=n_iter,
                              hop_length=hop_length,
                              win_length=n_fft,
                              window="hann")

def make_flag_audio(flag_text, sr=22050, duration=10.0):
    n_fft = 2048
    hop_length = 512
    time_frames = int(np.ceil(duration * sr / hop_length))
    img = text_to_image(flag_text, time_frames, 256, font_size=FONT_SIZE)
    mag = image_to_mag(img, n_fft=n_fft, hop_length=hop_length, max_amp=MAX_AMP)
    y = mag_to_audio(mag, n_fft=n_fft, hop_length=hop_length, n_iter=80)
    y = y / (np.max(np.abs(y)) + 1e-9)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr

def main():
    print("[+] Membuat stego audio...")
    stego, sr = make_flag_audio(FLAG, sr=SR, duration=FLAG_DURATION)

    host = AudioSegment.from_file(INPUT_MP3)
    tmp_wav = "._tmp_flag.wav"
    sf.write(tmp_wav, stego, sr, subtype="PCM_16")
    stego_seg = AudioSegment.from_wav(tmp_wav) + GAIN_DB
    os.remove(tmp_wav)

    mixed = host.overlay(stego_seg, position=int(START_TIME * 1000))
    mixed = mixed[:int(DURATION * 1000)]
    mixed.export(OUTPUT_WAV, format="wav")
    print("[+] Selesai! Cek spectrogram di", OUTPUT_WAV)

if __name__ == "__main__":
    main()

