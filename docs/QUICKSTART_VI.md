# Chạy MagicFace và xuất report AU/FER

File chạy chính: **`run_magicface.py`**. Config mẫu: **`configs/inference_demo.json`**.
Tutorial này chạy **inference và đánh giá output**; project chưa có chương trình training.

**Nếu muốn đưa vào một đường dẫn cho cả dataset**, dùng lệnh `dataset` theo
[DATASET_VI.md](DATASET_VI.md). Weights lấy trực tiếp từ Hugging Face; không cần
checkpoint tự train. Các bước dưới đây vẫn dùng được để cài môi trường.

Bạn có thể làm lần lượt: xem layout không GPU → sinh ảnh → chấm AU → thêm FER nếu có nhãn.
Các lệnh dưới đây dùng terminal Bash trên Linux/WSL, chạy tại thư mục gốc repo.
Có hai lựa chọn tạo môi trường: Conda hoặc `uv` (các bước cài bằng pip sau đó giống nhau).

## 1. Lấy code và tạo môi trường

Nếu chưa có repo:

```bash
git clone https://github.com/ngducmanh21/MagicFace.git
cd MagicFace
```

Nếu đã clone rồi, mở terminal trong `MagicFace` và cập nhật code:

```bash
git pull --ff-only
```

Tạo môi trường Python:

```bash
conda create -n magicface python=3.10 -y
conda activate magicface
python -m pip install -r requirements-report.txt
python run_magicface.py --help
```

CLI sẽ dùng đúng Python bạn đang gọi. Nếu chạy trong VS Code, chọn interpreter
`magicface` và kiểm tra terminal đã activate đúng môi trường.

**Nếu không có Conda nhưng đã có `uv`**, thay đoạn tạo môi trường ở trên bằng:

```bash
uv venv --python 3.10 --seed .venv
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python run_magicface.py --help
```

Với cách này, chọn `.venv/bin/python` trong VS Code. `uv` có thể tải Python nếu
máy chưa có phiên bản yêu cầu. Các lệnh `python -m pip` bên dưới vẫn giữ nguyên.

## 2. Xem thử report ngay, chưa cần GPU

```bash
python run_magicface.py preview
```

Mở file:

```text
runs/tutorial_preview/report.html
```

Lệnh này dùng ảnh mẫu có sẵn, lặp ảnh gốc ở cột kết quả để kiểm tra bố cục.
**Chưa sinh ảnh, chưa chấm AU, chưa đánh giá FER**: các giá trị `N/A` là đúng.
Không tải weights và không cần LibreFace.

Bạn có thể mở `report.html` trực tiếp bằng trình duyệt, hoặc xem
`runs/tutorial_preview/grid_001.png` trong VS Code.

## 3. Cài phần inference và kiểm tra GPU

Chỉ cần bước này khi muốn sinh ảnh mới. Script inference hiện dùng NVIDIA CUDA.

```bash
nvidia-smi
python -m pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-inference.txt
python -m pip check
python -c "import torch, numpy, diffusers; print('torch:', torch.__version__); print('numpy:', numpy.__version__); print('diffusers:', diffusers.__version__); print('CUDA:', torch.cuda.is_available())"
```

Cần thấy `CUDA: True` trước khi chạy inference. README gốc ghi khoảng 8 GB VRAM
cho inference; mức sử dụng thực tế phụ thuộc thiết lập và môi trường.

Dùng **`requirements-inference.txt` cho tutorial này**. File này pin NumPy 1.26.4
và Hugging Face Hub 0.25.2 để phù hợp bộ thư viện cũ của project. Không cài tiếp
`requirements.txt` gốc vào cùng môi trường vì nó pin NumPy 2.2.1.
Các nguồn kỹ thuật: [lệnh cài PyTorch 2.1.1](https://pytorch.org/get-started/previous-versions/#v211),
[Diffusers 0.25.1 sử dụng `cached_download`](https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/utils/dynamic_modules_utils.py),
[hướng dẫn lỗi ABI NumPy](https://numpy.org/doc/2.2/user/troubleshooting-importerror.html).

## 4. Sinh ba mức chỉnh sửa từ ảnh mẫu

Xem trước lệnh, không tải model hay tạo output:

```bash
python run_magicface.py run --config configs/inference_demo.json --no-au --dry-run
```

Chạy thật, tạm chưa chấm AU:

```bash
python run_magicface.py run --config configs/inference_demo.json --no-au
```

Config mẫu dùng:

- Ảnh gốc `test_images/00381.png` và background `test_images/00381_bg.png`.
- Hai AU theo thứ tự `AU4`, `AU1`.
- Ba mức `[0, 0]`, `[2, 1]`, `[4, 2]`.
- Seed `424` giữ nguyên giữa các mức, với 50 bước inference.

Model SD 1.5 và weights MagicFace sẽ được tải ở lần chạy đầu, nên cần mạng,
dung lượng ổ đĩa và thời gian tải. Lệnh sẽ in đường dẫn report khi hoàn tất.

```text
runs/demo_00381/
  generated/               # 3 ảnh mới từ model
  report.html              # mở file này để xem toàn bộ report
  grid_001.png             # ảnh trước/sau + bảng AU
  manifest.json
  results.json
  scores.csv
  summary.json
  run_config.json
  artifact_index.json
  figures/                 # các hình AU; N/A khi chưa chấm
  tables/                  # bảng số liệu dùng cho các hình
  images/                  # bản sao ảnh để chia sẻ report
```

`[0, 0]` là điều kiện không yêu cầu thay đổi AU, vẫn đi qua model sinh ảnh;
không đảm bảo output giống ảnh gốc từng pixel.

## 5. Chấm AU thật bằng LibreFace

Tạo môi trường **riêng** cho LibreFace, giữ môi trường `magicface` hiện tại:

```bash
conda create -n magicface-au python=3.9 -y
conda run -n magicface-au python -m pip install -r requirements-au.txt
conda run -n magicface-au python -c "import libreface; print('LibreFace import OK')"
MAGICFACE_AU_PYTHON="$(conda run -n magicface-au python -c 'import sys; print(sys.executable)')"
```

**Nếu dùng `uv` ở bước 1**, thay các lệnh Conda của bước này bằng:

```bash
uv venv --python 3.9 --seed .venv-au
.venv-au/bin/python -m pip install -r requirements-au.txt
.venv-au/bin/python -c "import libreface; print('LibreFace import OK')"
MAGICFACE_AU_PYTHON="$(pwd)/.venv-au/bin/python"
```

Không activate `.venv-au`; giữ `.venv` làm môi trường chính chạy CLI.
Truyền nguyên đường dẫn `.venv-au/bin/python`, không thay bằng target symlink
trong `~/.local/share/uv/python/`: Python gốc không thấy packages của virtualenv.

Nếu Dlib báo thiếu compiler/CMake khi cài, cần cài build tools của hệ điều hành
trước; tham khảo [hướng dẫn LibreFace](https://github.com/ihp-lab/LibreFace#-installation).
Weights của bộ chấm AU được tải ở lần chấm đầu tiên.

Chấm các ảnh đã sinh ở bước 4, **không sinh lại ảnh**:

```bash
python run_magicface.py report \
  --manifest runs/demo_00381/manifest.json \
  --output runs/demo_00381_scored \
  --au-python "$MAGICFACE_AU_PYTHON"
```

Mở **`runs/demo_00381_scored/report.html`**. Mỗi cặp ảnh có 12 cường độ AU đo
được trên ảnh gốc/kết quả và mức thay đổi. Kiểm tra dòng `AU scores: 3/3 cases
scored`; nếu thiếu score, xem nguyên nhân ngay trong report.

Nếu muốn sinh ảnh và chấm AU trong một lệnh ở các lần sau:

```bash
python run_magicface.py run --config configs/inference_demo.json \
  --output runs/demo_00381_full \
  --au-python "$MAGICFACE_AU_PYTHON"
```

LibreFace mặc định chạy CPU; inference vẫn dùng GPU. CLI kiểm tra import LibreFace
trước khi bắt đầu sinh ảnh khi bạn bật chấm AU. Nếu mở terminal mới, đặt lại biến
`MAGICFACE_AU_PYTHON` hoặc truyền thẳng đường dẫn Python của môi trường AU.

**Cách đọc:** `Request` là giá trị gửi vào MagicFace; `Source/Result` là AU đo
bởi LibreFace theo thang 0–5; `Change = Result - Source`. Không tự coi hai thang
đo bằng nhau. `Expected/Error` để `N/A` nếu chưa cung cấp hệ số quy đổi AU đã
được hiệu chuẩn. Chi tiết tại [hướng dẫn số liệu](verification.md#interpret-the-scores).

## 6. Đổi ảnh và các mức AU

Copy config để giữ lại ví dụ gốc:

```bash
cp configs/inference_demo.json configs/my_run.json
```

Sửa các trường sau trong `configs/my_run.json`:

```json
{
  "image": "../test_images/00512.png",
  "background": "../test_images/00512_bg.png",
  "output_dir": "../runs/my_run",
  "aus": ["AU6", "AU12"],
  "variations": [[0, 0], [1, 2], [2, 4], [-1, -2]],
  "seed": 424,
  "inference_steps": 50,
  "au_backend": "libreface",
  "au_device": "cpu",
  "figure_formats": ["png", "svg", "pdf"]
}
```

```bash
python run_magicface.py run --config configs/my_run.json --au-python "$MAGICFACE_AU_PYTHON"
```

Mỗi dòng `variations` phải có đúng số phần tử và đúng thứ tự trong `aus`.
Số âm và số thập phân đều được hỗ trợ. 12 AU hợp lệ:
`AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26`.
README gốc khuyến nghị điều khiển trong khoảng `[-10, 10]`.

Đường dẫn **trong config tính từ thư mục chứa config**; đường dẫn truyền qua
CLI tính từ terminal hiện tại. Có thể dùng đường dẫn tuyệt đối. Không đặt output
chung giữa các experiment nếu muốn giữ kết quả cũ; chạy lại cùng đường dẫn sẽ
thay các file cùng tên.

Ảnh riêng cần được chuẩn bị như ví dụ: ảnh RGB 512×512 có một khuôn mặt và ảnh
background/pose tương ứng. Hai ảnh phải cùng kích thước. Không dùng ảnh gốc làm
background thay thế. Phần tạo background cần thêm assets/preprocessing trong
[README gốc](../README.md#test-your-own-images); bộ cài inference ở tutorial này
chỉ phục vụ ảnh đã được chuẩn bị sẵn, không gồm InsightFace và các assets đó.

## 7. Xuất FER / confusion matrix

MagicFace không tự chạy bộ phân loại cảm xúc. Cần **nhãn thật và nhãn dự đoán**
từ FER model của bạn. Khi chưa có chúng, report ghi FER unavailable là đúng.

Thêm object `fer` vào từng phần tử trong `cases` của file
`runs/demo_00381_scored/results.json`. Ví dụ schema:

```json
"fer": {
  "source_true": "neutral",
  "source_pred": "neutral",
  "result_true": "happy",
  "result_pred": "happy"
}
```

Các nhãn trên chỉ minh họa, **không phải nhãn đã xác nhận của ảnh mẫu**. Thay
bằng nhãn thật/prediction của bạn. `result_true` cần được kiểm chứng trên ảnh
đã sinh, không lấy tự động từ cảm xúc mong muốn. Nếu không biết, để `null` hoặc
bỏ trường đó. Ảnh gốc lặp lại ở nhiều case phải có nhãn nhất quán.

Xuất lại report, giữ AU score đã tính và không gọi model:

```bash
python run_magicface.py report \
  --results runs/demo_00381_scored/results.json \
  --output runs/demo_00381_fer
```

Mở `runs/demo_00381_fer/report.html`. Nếu nhãn đầy đủ, sẽ có hình confusion matrix
cho source/result, accuracy và precision/recall/F1 theo lớp. Có cả số đếm và
matrix chuẩn hóa theo hàng. Tên hình và định nghĩa metrics nằm trong
[hướng dẫn evidence](verification.md#experiment-evidence-and-fer-evaluation).

## 8. Các lệnh hay dùng và lỗi thường gặp

```bash
python run_magicface.py run --help
python run_magicface.py report --help
python run_magicface.py report --results runs/demo_00381_scored/results.json --output runs/replot_png --formats png
python -m unittest discover -s tests -v
```

| Hiện tượng | Cách xử lý |
| --- | --- |
| `NVIDIA CUDA chua san sang` | Kiểm tra `nvidia-smi`, đúng môi trường `magicface`, bản PyTorch CUDA. Máy không GPU vẫn chạy `preview` và `report`. |
| `cached_download` / lỗi NumPy ABI | Cài lại theo bước 3 trong môi trường sạch; không trộn requirements gốc với bộ cài inference này. |
| Không import được LibreFace | Kiểm tra `--au-python` trỏ đúng Python trong `magicface-au`; xem stderr của lệnh import ở bước 5. |
| Không tìm thấy ảnh | Kiểm tra đường dẫn tương đối với vị trí config; `--dry-run` in các đường dẫn đã resolve. |
| Không phát hiện được mặt / AU score thiếu | Xem lỗi trong report; kiểm tra ảnh một mặt rõ ràng và việc tải AU weights. Không coi `N/A` là 0. |
| GPU hết VRAM | Đóng tác vụ chiếm GPU, thử thiết lập ảnh 512×512 như ví dụ. Chấm AU mặc định CPU. Giảm inference steps chủ yếu giảm thời gian, không giải quyết chắc chắn peak VRAM. |
| FER unavailable | Cần cung cấp cặp true/pred cho từng ảnh; không thể tính confusion matrix chỉ từ điều kiện AU. |

Đã kiểm tra CLI dry-run, đường dẫn, xuất report và số liệu bằng tests/fixtures.
Quy trình sinh ảnh và chấm AU thật vẫn cần được chạy trên môi trường có CUDA,
weights và dependencies tương ứng; tutorial không khẳng định đã benchmark model.
