# Chạy RAF-DB và AffectNet bằng pretrained MagicFace

Hai dataset dùng chung lệnh `dataset`, nhưng **bảng mã nhãn khác nhau**.
CLI đọc annotation và giữ nhãn ảnh nguồn trong output; không tự gán nhãn đó cho
ảnh sau chỉnh sửa. Weights vẫn lấy từ Hugging Face, không cần training trước.

## Lệnh chạy nhanh

Sau khi cài môi trường theo [QUICKSTART_VI.md](QUICKSTART_VI.md), cài thêm
`requirements-preprocess.txt` nếu chưa có background/pose 512×512. Đặt Python
chấm AU theo [DATASET_VI.md](DATASET_VI.md#1-cài-môi-trường-một-lần).

```bash
# RAF-DB: chỉ đọc test split, chưa tải model hay dùng GPU
python run_magicface.py dataset /data/RAF-DB \
  --dataset-type rafdb --split test --inspect

# Chạy thử 8 ảnh test -> 24 edit với config mặc định
python run_magicface.py dataset /data/RAF-DB \
  --dataset-type rafdb --split test --limit 8 --output runs/rafdb_test_8

# AffectNet: đọc validation split, chưa chạy model
python run_magicface.py dataset /data/AffectNet \
  --dataset-type affectnet --split val --inspect

python run_magicface.py dataset /data/AffectNet \
  --dataset-type affectnet --split val --limit 8 --output runs/affectnet_val_8
```

Thay `/data/...` bằng đường dẫn thật. `--split` được áp dụng **trước** `--limit`.
Bỏ `--limit` để chạy cả split. Chọn `--split train` nếu dùng ảnh train; mặc định
là `all` nếu không truyền split. RAF-DB Basic có train/test, không tự tạo val.
AffectNet CSV gốc có training/validation; bản repack có test chỉ được nhận nếu
thư mục/annotation của bản đó thực sự có test, không coi val là test.

## RAF-DB Basic

```text
RAF-DB/
  EmoLabel/list_patition_label.txt
  Image/original/train_00001.jpg
  Image/original/test_0001.jpg
  Image/aligned/train_00001_aligned.jpg
  Image/aligned/test_0001_aligned.jpg
```

Mỗi dòng TXT có dạng `train_00001.jpg 5`. Nhãn được đọc từ TXT; split lấy từ
prefix `train_`/`test_`. Nếu root có thêm tầng `basic/`, CLI cũng nhận được.
Tên file viết đúng thành `list_partition_label.txt` cũng được hỗ trợ.

`--raf-images auto` mặc định chọn toàn bộ cây `Image/aligned/` nếu thư mục đó
tồn tại, nếu không chọn `Image/original/`. Không đọc cả hai bản của một ảnh
thành hai mẫu. Với aligned, tên annotation `train_00001.jpg` được ghép với
`train_00001_aligned.jpg`. Bạn có thể chọn rõ bản ảnh:

```bash
python run_magicface.py dataset /data/RAF-DB --dataset-type rafdb \
  --raf-images original --split test --limit 8 --output runs/rafdb_original_8
```

Bạn cũng có thể truyền thẳng một thư mục ảnh nằm dưới `basic/Image/`, ví dụ
`basic/Image/aligned_224`. CLI sẽ tìm ngược lên `basic/EmoLabel/` và nhận đây là
ảnh aligned nhờ tên thư mục. Cả hai quy ước tên file đều được hỗ trợ:
`test_0001_aligned.jpg` và `test_0001.jpg`.

```bash
python run_magicface.py dataset /data/RAF-DB/basic/Image/aligned_224 \
  --dataset-type rafdb --split test --limit 8 --no-au \
  --output runs/rafdb_test_8
```

Ảnh 224×224 chưa phải input chuẩn 512×512 có background/pose, nên vẫn cần
`requirements-preprocess.txt`. `--no-au` chỉ bỏ bước LibreFace sau generation;
không bỏ preprocessing MagicFace.

Với metadata RAF aligned, preprocessing giữ nguyên crop khuôn mặt và chỉ resize
224→512 trước khi tạo background/pose. Nó không gọi detector để crop lần hai;
điều này tránh lỗi `The input image must contain a face` trên các crop sát mặt.

Nếu thấy `ModuleNotFoundError: libreface`, có hai cách:

```bash
# Cách nhanh: sinh ảnh trước, chưa đo AU
python run_magicface.py dataset /data/RAF-DB/basic/Image/aligned_224 \
  --dataset-type rafdb --split test --limit 8 --no-au \
  --output runs/rafdb_test_8

# Hoặc cài môi trường chấm AU riêng
uv venv --python 3.9 --seed .venv-au
.venv-au/bin/python -m pip install -r requirements-au.txt
python run_magicface.py dataset /data/RAF-DB/basic/Image/aligned_224 \
  --dataset-type rafdb --split test --limit 8 \
  --au-python "$PWD/.venv-au/bin/python" --output runs/rafdb_test_8_scored
```

Không nên chạy MagicFace bằng `.venv` của một project FER khác nếu môi trường đó
chưa cài `requirements-inference.txt` và `requirements-preprocess.txt`.
CLI kiểm tra các import preprocessing trước khi tải assets; nếu thiếu `insightface`
hoặc ONNX/OpenCV, generation chưa bắt đầu và output cũ không nên được tái sử dụng.

Nếu annotation/ảnh đặt ở chỗ khác:

```bash
python run_magicface.py dataset /data/RAF-DB --dataset-type rafdb \
  --annotations /data/labels/list_patition_label.txt \
  --image-root /data/raf_aligned --raf-images aligned --split test --inspect
```

Reader này dùng mã nhãn **Basic 1–7**. Nhãn 0 hoặc mã compound không được đoán
hay tự đổi offset; cần chuyển bản tùy biến sang [manifest generic](DATASET_VI.md#4-các-dạng-dataset-được-hỗ-trợ).
Ảnh aligned nhỏ hơn 512×512 vẫn đi qua preprocessing; upsample không tạo lại
chi tiết đã mất trong ảnh gốc. Dùng `original` khi cần giữ chất lượng ảnh đầu vào.

## AffectNet: ba kiểu lưu trữ

### CSV gốc

Nhận `training.csv`/`validation.csv` có cột `subDirectory_filePath` và `expression`.
Nếu có, đọc thêm `valence`, `arousal` và lưu bbox gốc trong metadata.

```text
AffectNet/
  Manually_Annotated/
    Manually_Annotated_Images/1/xxx.jpg
    file_lists/training.csv
    file_lists/validation.csv
  Automatically_Annotated/...
```

Cũng nhận CSV ở root dataset hoặc `Manually_Annotated_file_lists/`, với ảnh ở
`Manually_Annotated_Images/`. CLI chỉ tự tìm phần manually annotated.
Không trộn `Automatically_Annotated` vào ground truth. Nếu muốn dùng dự đoán
tự động, hãy tạo manifest generic với `source_pred` thay vì `source_true`.

Ví dụ nội dung CSV:

```csv
subDirectory_filePath,expression,valence,arousal
1/xxx.jpg,1,0.7,0.4
2/yyy.jpg,5,-0.5,0.3
```

Đường dẫn con lấy nguyên từ CSV. Với layout khác, truyền `--annotations` và
`--image-root` (thư mục chứa các đường dẫn tương đối trong CSV):

```bash
python run_magicface.py dataset /data/AffectNet --dataset-type affectnet \
  --annotations /data/lists/validation.csv \
  --image-root /data/Manually_Annotated_Images --split val --inspect
```

### NPY theo split

```text
AffectNet/
  train_set/
    images/000001.jpg
    annotations/000001_exp.npy
    annotations/000001_val.npy
    annotations/000001_aro.npy
  val_set/
    images/000001.jpg
    annotations/000001_exp.npy
    annotations/000001_val.npy
    annotations/000001_aro.npy
```

Đọc `*_exp.npy` làm nhãn categorical; `*_val.npy`/`*_aro.npy` là tùy chọn.
File phải chứa một giá trị số, không phải object/pickle. Cùng ID số ở train và
val vẫn là hai mẫu riêng. Có thể truyền root chung hoặc thẳng `train_set`/`val_set`;
`validation_set` và `test_set` cũng được nhận theo tên split.

### Thư mục lớp 0–7

```text
AffectNet/
  train/0/xxx.jpg
  train/1/yyy.jpg
  ...
  val/0/aaa.jpg
  val/7/bbb.jpg
```

**Phải truyền `--dataset-type affectnet`** để xác nhận sử dụng mapping AffectNet
0–7. Chế độ auto không đoán dataset chỉ từ các folder số, vì bản repack khác có
thể dùng thứ tự nhãn khác. Với folder classes không có annotation valence/arousal,
hai giá trị này để trống. Nếu bản của bạn đã remap class IDs, dùng manifest
generic với tên cảm xúc rõ ràng.

## Mapping và lọc lớp

Tên lớp nội bộ được chuẩn hóa để dùng chung khi visualize:

| Tên nội bộ | RAF-DB Basic | AffectNet |
| --- | --- | --- |
| `neutral` | 7 | 0 |
| `happy` | 4 | 1 |
| `sad` | 5 | 2 |
| `surprise` | 1 | 3 |
| `fear` | 2 | 4 |
| `disgust` | 3 | 5 |
| `anger` | 6 | 6 |
| `contempt` | Không có | 7 |

Mapping RAF dựa trên định dạng Basic bạn cung cấp và [paper RAF-DB](https://www.whdeng.cn/RAF/li_RAFDB_2017_CVPR.pdf).
Mapping AffectNet 0–7 cũng được ghi trong [implementation EmoNet của tác giả](https://github.com/face-analysis/emonet#class-number-to-expression-name).

AffectNet mặc định giữ 8 lớp 0–7. Các mã 8/9/10 không thuộc tập emotion được
đánh giá ở đây và bị loại, có lý do trong `excluded_annotations.csv`. Chọn bản 7 lớp:

```bash
python run_magicface.py dataset /data/AffectNet --dataset-type affectnet \
  --affectnet-classes 7 --split val --inspect
```

Lúc đó contempt (7) cũng bị loại, không đổi mã của các lớp còn lại.
Annotation malformed được ghi riêng; ảnh thiếu/hỏng được ghi ở `failures.csv`.
Valence/arousal ngoài `[-1, 1]`, sentinel `-2` hoặc thiếu dữ liệu được để null/
ô trống, không thay bằng 0. Không tự tìm/tải hai dataset: bạn dùng bản local đã có.

## Thông số để visualize

Output vẫn có grid, ảnh sinh, AU scores và report chung. Bổ sung:

- `samples.csv`, `scores.csv`: `dataset_name`, `dataset_split`, `source_emotion`,
  `source_valence`, `source_arousal`, kèm seed/steps/thời gian và AU như trước.
- `dataset_inputs.json`: nhãn ID gốc, nguồn annotation, đường dẫn ảnh và split.
- `dataset_summary.json`: số mẫu được chọn theo emotion và split; số annotation bị loại.
- `excluded_annotations.csv`: mẫu bị loại khỏi tập lớp hoặc annotation lỗi.
- `tables/source_label_groups.csv`: số nguồn duy nhất, số edit và số cặp có AU score theo lớp/split.
- `tables/au_by_source_emotion.csv`: trung bình AU trước/sau và thay đổi, nhóm theo nhãn **ảnh nguồn**.
- `figures/fig_source_emotion_counts.*`: phân bố nhãn nguồn trong các mẫu đã có ảnh sinh.

Biểu đồ nhãn nguồn đếm mỗi source path một lần, không nhân số mẫu lên theo số
biến thể AU. Bảng trung bình AU vẫn tính theo cặp edit hợp lệ như report trước;
các AU combinations được gộp mô tả, không phải tác động độc lập của một AU.
Mẫu lỗi không xuất hiện trong hình phân bố output; xem coverage và dataset summary
để so với số input đã chọn. CSV có thể join bằng `sample_id`.

Nhãn RAF/AffectNet được đưa vào **`fer.source_true`**. Không tự điền `source_pred`,
`result_true` hay `result_pred`. Confusion matrix cần thêm prediction của FER model;
ảnh đã chỉnh sửa cần được kiểm chứng lại nhãn. Khi điền predictions, dùng cùng
tên lớp nội bộ trong bảng trên để tránh coi `Happiness` và `happy` là hai lớp.

## Phạm vi đã kiểm tra

Đã kiểm tra mapping, split, đường dẫn original/aligned, CSV nested, NPY, folder
classes, loại annotation và metadata/report bằng fixtures. Chưa có hai dataset
thật trong workspace để kiểm tra số lượng thực tế, và chưa chạy model trên GPU.
