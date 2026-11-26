import numpy as np
import matplotlib.pyplot as plt

# 1. Doc file .bin va file MFCC tu the SD
pcm = np.fromfile("i2s_32k_10s.bin", dtype=np.int16).astype(np.float32)
stm32_mfcc = np.loadtxt("i2s_32k_mfcc.txt", delimiter=",")[:, 1:] # Bo cot frame_idx

# 2. Tinh MFCC doc lap tren Python theo dung thong so mfcc.c
fs, n_fft, hop, n_mels, n_mfcc = 32000, 1024, 512, 40, 13
bins = n_fft // 2 + 1
win = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1))

mel_low = 2595.0 * np.log10(1.0 + 100.0 / 700.0)
mel_high = 2595.0 * np.log10(1.0 + 7000.0 / 700.0)
mel_pts = np.linspace(mel_low, mel_high, n_mels + 2)
hz_pts = 700.0 * (10.0**(mel_pts / 2595.0) - 1.0)
edges = np.minimum(np.floor((n_fft + 1) * hz_pts / fs).astype(int), bins - 1)

fb = np.zeros((n_mels, bins), dtype=np.float32)
for m in range(n_mels):
  l, c, r = edges[m], edges[m + 1], edges[m + 2]
  if c != l:
    for k in range(l, min(c, bins)):
      fb[m, k] = (k - l) / (c - l)
  if r != c:
    for k in range(c, min(r + 1, bins)):
      fb[m, k] = (r - k) / (r - c)

n_idx = np.arange(n_mfcc)[:, None]
m_idx = np.arange(n_mels)[None, :]
dct = np.cos(np.pi * n_idx * (2 * m_idx + 1) / (2.0 * n_mels)).astype(np.float32)

n_frames = 1 + (len(pcm) - n_fft) // hop
py_mfcc = np.zeros((n_frames, n_mfcc), dtype=np.float32)
for i in range(n_frames):
  frame = pcm[i * hop : i * hop + n_fft] * (1.0 / 32768.0)
  frame = (frame - np.mean(frame)) * win
  spec = np.fft.rfft(frame)
  power = (spec.real**2 + spec.imag**2) / n_fft
  mel_e = 20.0 * np.log10(np.maximum(fb @ power, 1e-6))
  py_mfcc[i] = dct @ mel_e

# 3. In sai so giua STM32 va Python
diff = np.abs(stm32_mfcc - py_mfcc)
print(f"So khung: STM32={len(stm32_mfcc)}, Python={len(py_mfcc)}")
print(f"Sai so lon nhat (Max Abs Error): {diff.max():.4f}")
print(f"Sai so trung binh (Mean Abs Error): {diff.mean():.4f}")
print("STM32 frame 0:", np.round(stm32_mfcc[0], 2))
print("Python frame 0:", np.round(py_mfcc[0], 2))
print("Sai so TB tung cot (c0..c12):", np.round(diff.mean(axis=0), 2))
print("So mau doc duoc len(pcm):", len(pcm))
print("He so tuong quan c0 (STM32 vs Python):", round(float(np.corrcoef(stm32_mfcc[:, 0], py_mfcc[:, 0])[0, 1]), 4))
print("Frame 184 c0 -> STM32:", stm32_mfcc[184, 0], "| Python:", round(float(py_mfcc[184, 0]), 2))

# 4. Ve anh Heatmap MFCC (c1..c12) de quan sat su kien am thanh theo thoi gian
plt.figure(figsize=(12, 5))
plt.imshow(stm32_mfcc[:, 1:].T, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Gia tri MFCC")
plt.xlabel("Frame (0..623 = 10 giay)")
plt.ylabel("He so MFCC (c1..c12)")
plt.title("Bieu do nhiet MFCC thu tu STM32")
plt.tight_layout()
plt.show()