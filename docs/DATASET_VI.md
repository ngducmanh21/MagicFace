# Một đường dẫn dataset → ảnh chỉnh sửa + report AU/FER

**RAF-DB / AffectNet:** xem [hướng dẫn riêng](RAF_AFFECTNET_VI.md) để đọc TXT,
CSV/NPY hoặc folder classes đúng mapping và chọn split.

CLI này dùng weights **[mengtingwei/magicface trên Hugging Face](https://huggingface.co/mengtingwei/magicface)**, với SD 1.5
làm base. Không cần train hay tự tạo checkpoint trước.

Sau khi cài môi trường, lệnh chính là:

```bash
python run_magicface.py dataset /duong/dan/dataset
```

Mỗi dataset là một lần chạy. CLI tìm ảnh, ghép background có sẵn hoặc chuẩn bị
ảnh gốc, nạp MagicFace một lần, chạy từng mức AU, rồi gom kết quả vào một report.
Mặc định dùng ba mức AU4/AU1: `[0,0]`, `[2,1]`, `[4,2]`, seed 424 và 50 steps.

## 1. Cài môi trường một lần

Làm bước 1, 3 và 5 trong [QUICKSTART_VI.md](QUICKSTART_VI.md) để có môi trường
inference CUDA và môi trường chấm AU riêng. Nếu dataset chỉ có ảnh gốc, cài thêm
trong môi trường inference:

```bash
python -m pip install -r requirements-preprocess.txt
```

Dataset đã có ảnh crop/background thì không cần gói preprocessing này.
InsightFace có thể cần compiler khi cài từ source; trên Linux server dùng bộ
OpenCV headless đã pin trong requirements. Bộ dependencies đã được kiểm tra
resolve, nhưng không thay thế việc kiểm tra CUDA/weights trên máy chạy thật.

Đặt Python chấm AU một lần cho terminal hiện tại. Nếu tạo môi trường bằng `uv`:

```bash
export MAGICFACE_AU_PYTHON="$PWD/.venv-au/bin/python"
```

Nếu dùng Conda:

```bash
export MAGICFACE_AU_PYTHON="$(conda run -n magicface-au python -c 'import sys; print(sys.executable)')"
```

Có thể truyền `--au-python /duong/dan/python` hoặc đặt `au_python` trong config
thay vì biến môi trường. Thứ tự ưu tiên: CLI → config → biến môi trường → Python
đang chạy CLI. Inference cần NVIDIA CUDA; bộ chấm AU mặc định chạy CPU.

## 2. Kiểm tra dataset trước

```bash
python run_magicface.py dataset /duong/dan/dataset --inspect
```

Lệnh này không cần GPU, không tải model và không ghi output. Nó in số ảnh đã
tìm thấy, số ảnh có background, số ảnh cần preprocess, số input lỗi và số edit
dự kiến. `--dry-run` chỉ in lệnh; `--inspect` thực sự đọc/kiểm tra ảnh.

Thử với ảnh mẫu đã có trong repo:

```bash
python run_magicface.py dataset test_images \
  --config configs/inference_demo.json --limit 3 --inspect
```

Kết quả mong đợi là **3 ảnh gốc + 3 background, dự kiến 9 ảnh sinh**. Config
demo này chỉ để kiểm tra plumbing; run Anger chính dùng audited default config.

## 3. Chạy dataset

Nên thử vài ảnh trước để kiểm tra môi trường và thời gian xử lý:

```bash
python run_magicface.py dataset /duong/dan/dataset --limit 8 --output runs/check_8_images
```

Chạy toàn bộ:

```bash
python run_magicface.py dataset /duong/dan/dataset --output runs/dataset_full
```

`--limit` tính theo **cell-source pair** sau bước audited selection khi config có
`cell_selection`; explicit `--limit 8` là smoke test 8 pair đầu tiên.
Bỏ `--output` thì CLI
tạo thư mục mới dưới `runs/`, mang tên dataset và timestamp UTC. Thư mục output
phải mới/rỗng để không trộn kết quả của các lần chạy.

Nếu muốn sinh ảnh và lưu thông số trước, chưa chấm AU:

```bash
python run_magicface.py dataset /duong/dan/dataset --limit 8 --no-au --output runs/images_first
```

Sau đó chấm các ảnh đã sinh, không chạy lại diffusion:

```bash
python run_magicface.py report --manifest runs/images_first/generation_manifest.json \
  --output runs/images_scored --au-python "$MAGICFACE_AU_PYTHON"
```

## 4. Các dạng dataset được hỗ trợ

**Thư mục ảnh gốc:** scan cả thư mục con, hỗ trợ JPG/JPEG/PNG/WebP/BMP/TIFF.

```text
my_dataset/
  person_a/001.jpg
  person_b/001.jpg
  another.png
```

Ảnh thiếu background sẽ được crop về 512×512 và tạo background/pose bằng các
script gốc của MagicFace. Assets bổ trợ được tải từ cùng repo Hugging Face ở lần
đầu; chỉ tải các phần preprocessing, không tải lặp lại toàn bộ model. Giai đoạn
này chạy trong process riêng và kết thúc trước khi nạp diffusion để giải phóng VRAM.
Ảnh có nhiều mặt dùng mặt lớn nhất; nên dùng ảnh một người nếu có nhãn cảm xúc.

**Ảnh đã chuẩn bị:** ghép `name.png` với `name_bg.png` trong cùng thư mục,
hoặc dùng hai cây thư mục song song:

```text
my_dataset/
  images/person_a/001.jpg
  backgrounds/person_a/001.png
```

Background cũng có thể tên `001_bg.png`. Cả source/background đã chuẩn bị phải
là 512×512. Không tự resize hoặc lấy ảnh gốc giả làm background. Khi có nhiều
background cùng khớp một ảnh, input được ghi lỗi để bạn chỉ định rõ trong manifest.
`--prepared-only` bỏ qua các ảnh thiếu background thay vì tự preprocess.

Ảnh `_bg` và các thư mục `backgrounds`, `prepared`, `generated`, `runs`, thư mục
ẩn không bị scan thành ảnh gốc. Nếu có thư mục `images/`, chỉ scan cây đó.
Tên ảnh trùng ở các thư mục con được giữ riêng bằng ID và thư mục output riêng.

**Manifest JSON:** truyền thẳng đường dẫn JSON, hoặc đặt tên `dataset.json`
ở gốc dataset để CLI tự nhận. Xem file chạy được [examples/dataset.json](../examples/dataset.json).

```json
{
  "metadata": {"dataset": "my_dataset", "split": "test"},
  "items": [
    {
      "id": "sample_001",
      "source": "images/001.jpg",
      "background": "backgrounds/001.png",
      "metadata": {"subject_id": "person_a"}
    }
  ]
}
```

Bỏ trường `background` khi ảnh chưa được preprocess. Đường dẫn trong manifest
tính từ vị trí file manifest. ID phải duy nhất. Manifest dữ liệu dùng **`items`**;
manifest report dùng **`cases`** và chạy bằng lệnh `report`, không phải `dataset`.

**Manifest CSV:** cùng quy tắc đường dẫn, các cột `source`, `background`, `id`:

```csv
id,source,background,split
sample_001,images/001.jpg,backgrounds/001.png,test
sample_002,raw/002.jpg,,test
```

`id` và `background` có thể bỏ trống; cột bổ sung được giữ trong metadata.
Nếu có cả `dataset.json` và `dataset.csv` trong cùng thư mục, hãy truyền thẳng
file cần dùng để tránh chọn nhầm.

## 5. Lấy thông số để visualize

```text
runs/dataset_full/
  report.html                 # report chung, bao gồm coverage và lỗi input
  generated/                  # ảnh sinh; tách theo ID để tránh ghi đè
  prepared/                   # source crop/background được tạo từ ảnh gốc
  samples.csv                 # một dòng mỗi ảnh sinh
  samples.jsonl               # ghi nối tiếp ngay sau mỗi edit thành công
  scores.csv                  # một dòng mỗi ảnh sinh × mỗi AU
  results.json                # số đo đầy đủ + metadata từng sample
  dataset_inputs.json         # snapshot danh sách input đã chọn
  cell_selection.json         # snapshot audited rare-cell allowlist nếu dùng
  scope_audit.json             # checksum, exact allowed pairs, no-random-fallback contract
  dataset_summary.json        # số lượng, trạng thái, thời gian, thông số chạy
  generation_manifest.json    # dùng chấm lại AU mà không sinh lại ảnh
  failures.csv / failures.json
  grid_001.png, ...
  figures/                    # intensity, control response, heatmap AU; FER nếu có nhãn
  tables/                     # các bảng số liệu tổng hợp
```

Các trường chính trong `samples.csv`: `sample_id`, `dataset_id`, `input_source`,
`source`, `background`, `result`, `requested_aus`, `seed`, `inference_steps`,
`generation_seconds`. Thời gian này đo lời gọi sinh một ảnh, gồm decode/postprocess,
không gồm tải model, tiền xử lý dataset hay chấm AU; không phải benchmark GPU độc lập.

`scores.csv` bổ sung AU source/result, measured delta, expected delta và sai số
nếu đã cấu hình hệ số quy đổi. Có thể join hai CSV bằng `sample_id`.
`dataset_summary.json` lưu model IDs, seed, các mức AU, số bước, số edit dự kiến/
thành công/chưa sinh và lỗi. Config sửa tại [configs/dataset_demo.json](../configs/dataset_demo.json):

```bash
python run_magicface.py dataset /duong/dan/dataset --config configs/dataset_demo.json
```

Config mặc định đọc allowlist `anger_cells_magicface.json`: 5 rare Anger cells,
38 cell-source pairs từ 32 nguồn train duy nhất. Mỗi pair có một `zero_baseline`
và target AU5 hoặc AU25 riêng lẻ ở mức +1…+4, tổng 190 outputs.
`single_au_only` và `require_zero_baseline` khiến CLI từ chối config vô tình ghép
nhiều AU hoặc thiếu/thừa baseline. Seed, prompt và inference steps giữ cố định.
Không tự bù lên 50 bằng ảnh ngoài rule. Muốn mở rộng emotion/cell phải tạo allowlist
mới qua cùng R1–R10, không dùng thứ tự ảnh hoặc sampling ngẫu nhiên.

Giá trị source/result là AU đo thực tế; tham số request chỉ là điều kiện đầu vào.
Thiếu score được để `N/A`, không thay bằng 0. Chưa hiệu chuẩn thang AU thì giữ
`au_delta_scale: null`. Xem [định nghĩa metrics](verification.md#interpret-the-scores).

Ảnh lỗi được ghi riêng và các ảnh hợp lệ tiếp tục chạy. Nếu một ảnh đã sinh được
một số edit rồi lỗi, những edit thành công vẫn được giữ. `samples.jsonl` được ghi
sau mỗi edit; manifest tổng được cập nhật định kỳ và khi kết thúc/bị ngắt có xử lý.
Chưa có tự động resume; dùng thư mục output mới khi chạy lại.

## 6. FER và dữ liệu nhãn

Chế độ generic không suy ra nhãn cảm xúc từ tên thư mục; ngoại lệ được khai báo
rõ là `--dataset-type affectnet` cho folder class 0–7. Không coi AU yêu cầu là ground truth.
Input JSON có thể thêm `fer: {"source_true": "happy", "source_pred": "happy"}`;
CSV có thể thêm hai cột `source_true`, `source_pred`. Đây chỉ là ví dụ schema:
hãy dùng nhãn thật và dự đoán của bạn. Không gán nhãn của ảnh gốc cho mọi ảnh sinh.

Sau khi có prediction/ground truth cho các ảnh sinh, thêm `result_true` và
`result_pred` vào từng `cases[].fer` trong `results.json`, rồi chạy:

```bash
python run_magicface.py report --results runs/dataset_full/results.json --output runs/dataset_fer
```

Lệnh này giữ nguyên AU score, không gọi lại model, và xuất confusion matrix/F1
khi đủ dữ liệu. Thiếu nhãn thì FER unavailable.

## Phạm vi đã kiểm tra

Đã kiểm tra scan/ghép dữ liệu, lỗi ảnh, model load một lần bằng test double,
giữ kết quả khi lỗi giữa chừng, metadata/CSV và report. Trên máy hiện tại chưa chạy
pipeline Hugging Face/preprocessing/AU thật do chưa có CUDA và weights. `--inspect`
có thể dùng ngay trong môi trường chỉ cài `requirements-report.txt`.
