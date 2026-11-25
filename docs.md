# Thư viện MFCC C99 cho Hệ thống Nhúng và TinyML

Thư viện trích xuất đặc trưng âm thanh **MFCC (Mel-Frequency Cepstral Coefficients)** viết bằng ngôn ngữ C99 thuần túy, không phụ thuộc thư viện ngoài (như CMSIS-DSP) và **không cấp phát động (`malloc`/`free`)**. Toàn bộ bộ nhớ làm việc được quản lý thông qua một vùng nhớ tĩnh (`workspace`) do người dùng cung cấp, giúp triển khai đồng nhất trên cả PC (Python `ctypes`) và vi điều khiển (STM32, ESP32).

## Tổng quan về MFCC và quy trình xử lý

MFCC mô phỏng cách cảm nhận âm thanh phi tuyến tính của tai người (nhạy ở dải tần thấp và giảm dần độ phân giải ở dải tần cao), giúp nén dữ liệu sóng âm thành một ma trận đặc trưng nhỏ gọn phù hợp cho các mô hình TinyML như **MLP** hoặc **1D-CNN**.

Quy trình xử lý cho mỗi khung tín hiệu (frame) dài `n_fft` mẫu trong thư viện:

1. **Chuẩn hóa biên độ và khử DC (`DC Removal`)**

   - Nhân từng mẫu với `input_scale`.
   - Nếu `dc_removal != 0`, tính giá trị trung bình của khung và trừ đi để loại bỏ độ lệch một chiều (DC offset) từ micro (như INMP441).

2. **Nhân cửa sổ Hamming (`Windowing`)**

   - Giảm hiện tượng rò rỉ phổ (spectral leakage) tại hai đầu khung bằng hàm: `w[i] = 0.54 - 0.46 * cos(2 * pi * i / (n_fft - 1))`.

3. **Biến đổi FFT (`fft_reference`)**

   - Chuyển đổi khung tín hiệu từ miền thời gian sang miền tần số bằng thuật toán Radix-2 Cooley-Tukey tại chỗ.

4. **Tính phổ công suất (`Power Spectrum`)**

   - Chỉ giữ lại `bins = n_fft / 2 + 1` bin tần số thực không đối xứng (từ `0 Hz` đến tần số Nyquist `fs_hz / 2`).
   - Công suất tại bin `k`: `power[k] = (real^2 + imag^2) / n_fft`.
   - Độ phân giải mỗi bin tần số là `delta_f = fs_hz / n_fft` (với `32 kHz` và `1024` mẫu thì mỗi bin cách nhau `31.25 Hz`).

5. **Áp dụng dãy lọc tam giác Mel (`Mel Filterbank`)**

   - Quy đổi tần số Hz sang thang Mel: `mel = 2595 * log10(1 + hz / 700)`.
   - Gom năng lượng từ `bins` bin tần số vào `n_mels` dải lọc tam giác cách đều trên thang Mel trong khoảng `[fmin_hz, fmax_hz]`.

6. **Nén thang Logarit (`Log Compression`)**

   - Tính năng lượng log: `mel_energy[m] = log_scale * log10(max(sum, log_floor))`.

7. **Biến đổi Cosine rời rạc (`Type-II DCT`)**

   - Chiếu vector `n_mels` chiều sang không gian cepstral và giữ lại `n_mfcc` hệ số đầu tiên làm vector đặc trưng đầu ra.

## Tài liệu API

Để sử dụng thư viện, include file header `#include "mfcc.h"` và liên kết với thư viện toán học (`-lm` nếu biên dịch bằng GCC).

### Cấu trúc cấu hình `mfcc_config_t`

| Trường | Kiểu dữ liệu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `fs_hz` | `uint32_t` | `32000U` | Tần số lấy mẫu của tín hiệu (Hz) |
| `n_fft` | `uint32_t` | `1024U` | Kích thước khung FFT (phải là lũy thừa của 2) |
| `hop` | `uint32_t` | `512U` | Bước trượt giữa hai khung liên tiếp (mẫu) |
| `n_mels` | `uint32_t` | `40U` | Số lượng bộ lọc tam giác Mel |
| `n_mfcc` | `uint32_t` | `13U` | Số lượng hệ số MFCC đầu ra mỗi khung |
| `fmin_hz` | `float` | `20.0f` | Tần số thấp nhất của dải lọc Mel (Hz) |
| `fmax_hz` | `float` | `16000.0f` | Tần số cao nhất của dải lọc (`<= fs_hz / 2`) |
| `log_floor` | `float` | `1.0e-6f` | Ngưỡng chặn dưới tránh lỗi `log10(0)` |
| `log_scale` | `float` | `20.0f` | Hệ số nhân sau phép tính `log10` |
| `input_scale` | `float` | `1.0f` | Hệ số chuẩn hóa đầu vào |
| `window_frames` | `uint32_t` | `1U` | Tham số dự phòng cho tầng gom khung ngoài |
| `dc_removal` | `uint32_t` | `1U` | Bật (`1`) hoặc tắt (`0`) khử DC theo khung |

### Mã trạng thái `mfcc_status_t`

- `MFCC_OK`: Thực thi thành công.
- `MFCC_ERROR_ARGUMENT`: Con trỏ đầu vào (`NULL`) không hợp lệ.
- `MFCC_ERROR_CONFIG`: Tham số trong `mfcc_config_t` vi phạm ràng buộc hợp lệ.
- `MFCC_ERROR_WORKSPACE`: Bộ nhớ `workspace` không đủ kích thước hoặc không căn lề theo `sizeof(float)` (4 byte).
- `MFCC_ERROR_OUTPUT`: Mảng `output` không đủ chỗ chứa số khung cần xuất.

### Danh sách hàm

#### `mfcc_default_config`

```c
void mfcc_default_config(mfcc_config_t *config);
```

Khởi tạo cấu trúc `mfcc_config_t` với bộ tham số mặc định cho âm thanh `32 kHz` (`n_fft = 1024`, `hop = 512`, `n_mels = 40`, `n_mfcc = 13`).

Tham số:

- `config`: Con trỏ tới struct `mfcc_config_t` cần điền giá trị mặc định.

#### `mfcc_workspace_size`

```c
size_t mfcc_workspace_size(const mfcc_config_t *config);
```

Tính chính xác tổng số byte RAM cần cấp cho vùng nhớ `workspace` dựa trên cấu
hình đã chọn.

Tham số:

- `config`: Con trỏ tới cấu hình MFCC.

Giá trị trả về:

- Số byte cần thiết (đã căn lề bội của 4 byte), hoặc `0` nếu `config` không hợp lệ.

#### `mfcc_init`

```c
mfcc_status_t mfcc_init(
  mfcc_context_t *context,
  const mfcc_config_t *config,
  void *workspace,
  size_t workspace_size
);
```

Phân chia vùng nhớ `workspace` cho các bộ đệm nội bộ và tính sẵn bảng cửa sổ
Hamming, ma trận bộ lọc Mel và ma trận DCT.

Tham số:

- `context`: Con trỏ tới đối tượng ngữ cảnh `mfcc_context_t`.
- `config`: Con trỏ tới cấu hình MFCC đã thiết lập.
- `workspace`: Con trỏ tới vùng RAM tĩnh do người dùng cấp (phải chia hết cho 4 byte).
- `workspace_size`: Kích thước thực tế (byte) của vùng `workspace`.

Giá trị trả về:

- `MFCC_OK` nếu thành công, hoặc mã lỗi `MFCC_ERROR_*` tương ứng.

#### `mfcc_compute_frame`

```c
mfcc_status_t mfcc_compute_frame(mfcc_context_t *context,
                                 const float *samples,
                                 float *output);
```

Tính vector MFCC cho đúng 1 khung tín hiệu có độ dài `config.n_fft` mẫu.

Tham số:

- `context`: Ngữ cảnh đã được khởi tạo bởi `mfcc_init`.
- `samples`: Mảng đầu vào chứa ít nhất `n_fft` phần tử kiểu `float`.
- `output`: Mảng đầu ra chứa tối thiểu `n_mfcc` phần tử kiểu `float`.
Giá trị trả về:
- `MFCC_OK` hoặc `MFCC_ERROR_ARGUMENT`.

#### `mfcc_compute_frames`

```c
mfcc_status_t mfcc_compute_frames(mfcc_context_t *context,
                                  const float *samples,
                                  size_t sample_count,
                                  float *output,
                                  size_t output_frames,
                                  size_t *written_frames);
```

Trượt cửa sổ theo bước `hop` trên toàn bộ mảng tín hiệu dài `sample_count` mẫu
và xuất liên tiếp các vector MFCC vào `output`.

Tham số:

- `context`: Ngữ cảnh đã khởi tạo.
- `samples`: Mảng tín hiệu âm thanh đầu vào.
- `sample_count`: Tổng số mẫu trong `samples`.
- `output`: Mảng 1 chiều lưu liên tiếp `frames * n_mfcc` giá trị `float`.
- `output_frames`: Số khung tối đa mà mảng `output` có thể chứa.
- `written_frames`: Con trỏ nhận số khung thực tế đã tính (có thể truyền `NULL` nếu không dùng).

Giá trị trả về:

- `MFCC_OK`, `MFCC_ERROR_ARGUMENT` hoặc `MFCC_ERROR_OUTPUT`.

## Ví dụ sử dụng nhanh

Ví dụ dưới đây trích xuất MFCC cho `1 giây` âm thanh `32 kHz` (`32,000` mẫu `int16_t` đã thu vào RAM), cho ra ma trận `61 x 13` phục vụ suy luận TinyML:

```c
#include "mfcc.h"
#include <stdint.h>

#define AUDIO_SAMPLES 32000U
#define MAX_FRAMES    61U
#define NUM_MFCC      13U

/* Dat workspace vao AXI SRAM (RAM_D1) neu dung STM32H7 */
__attribute__((aligned(4))) static uint8_t mfcc_ws[110 * 1024];
static mfcc_context_t mfcc_ctx;
static float audio_f32[AUDIO_SAMPLES];
static float mfcc_features[MAX_FRAMES * NUM_MFCC];

int run_mfcc_pipeline(const int16_t *pcm16_buf)
{
  mfcc_config_t cfg;
  mfcc_default_config(&cfg);
  cfg.input_scale = 1.0f / 32768.0f; /* Neu model train voi WAV chuan hoa [-1, 1] */

  size_t needed = mfcc_workspace_size(&cfg);
  if (needed == 0U || needed > sizeof(mfcc_ws)) {
    return -1;
  }

  if (mfcc_init(&mfcc_ctx, &cfg, mfcc_ws, sizeof(mfcc_ws)) != MFCC_OK) {
    return -1;
  }

  for (uint32_t i = 0U; i < AUDIO_SAMPLES; ++i) {
    audio_f32[i] = (float)pcm16_buf[i];
  }

  size_t written = 0U;
  mfcc_status_t st = mfcc_compute_frames(
    &mfcc_ctx, audio_f32, AUDIO_SAMPLES,
    mfcc_features, MAX_FRAMES, &written
  );
  if (st != MFCC_OK) {
    return -1;
  }

  /* mfcc_features bay gio chua (written x 13) dac trung cho MLP / 1D-CNN */
  return (int)written;
}
```

## Các lưu ý quan trọng khi triển khai

### Dung lượng bộ nhớ RAM

- Với cấu hình mặc định (`n_fft = 1024`, `n_mels = 40`, `n_mfcc = 13`), kích thước `mfcc_workspace_size` cần khoảng **`102,756 bytes` (~`100.35 KB`)**, trong đó riêng ma trận `mel_filterbank` (`40 x 513`) chiếm **`82 KB`**.
- **Trên STM32H7:** Tránh đặt `mfcc_ws` vào `DTCMRAM` (`128 KB`) nếu đã chứa FatFs hoặc buffer thu âm; hãy đặt vào vùng **AXI SRAM (`0x24000000`)**.
- **Trên MCU RAM nhỏ (STM32F4, ESP32):** Có thể giảm `n_fft = 512U`, `hop = 256U`, `n_mels = 26U` để giảm `workspace` xuống còn khoảng `~34 KB`.

### Đồng bộ tuyệt đối giữa lúc Train Python và chạy trên MCU

1. **Hệ số `input_scale`:**

   - Nếu khi huấn luyện AI trên Python bạn chuẩn hóa dữ liệu từ `int16_t` về `[-1.0, +1.0]` (chia `32768.0`), trên MCU bắt buộc đặt `cfg.input_scale = 1.0f / 32768.0f`.
   - Nếu truyền trực tiếp giá trị `float` đã chia `32768.0f` thì giữ `cfg.input_scale = 1.0f`.

2. **Khác biệt công thức so với thư viện `librosa`:**

   - Thư viện này sử dụng thang Mel HTK (`2595 * log10(1 + f / 700)`), làm tròn chỉ số bin nguyên (`floorf`), tính `log_scale = 20.0f` trên phổ công suất (`power`), và DCT không chuẩn hóa trực giao.
   - Để tránh lệch đặc trưng làm tụt độ chính xác sau lượng tử hóa `INT8` (PTQ), khuyến nghị biên dịch chính file `mfcc.c` này thành thư viện động (`.dll` / `.so`) và gọi qua Python `ctypes` khi tạo tập dữ liệu huấn luyện.

### Điều kiện hợp lệ của tham số cấu hình

- `n_fft` bắt buộc là lũy thừa của 2 và lớn hơn hoặc bằng `2`.
- `n_mels + 2 <= n_fft` và `n_mfcc <= n_mels`.
- `0.0f <= fmin_hz < fmax_hz <= fs_hz * 0.5f`.
- Vùng nhớ `workspace` truyền vào `mfcc_init` bắt buộc phải được căn lề theo
  4 byte (`__attribute__((aligned(4)))` hoặc mảng kiểu `float`).
- Trường `window_frames` hiện được gán mặc định bằng `1U` và không tham gia
  tính toán bên trong `mfcc.c`; nó được giữ lại cho tầng quản lý cửa sổ khung
  ở ứng dụng phía trên.
