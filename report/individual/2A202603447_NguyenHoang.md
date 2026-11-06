# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603447
- **Họ và tên:** Nguyễn Hoàng
- **SEED** (in ở ô đầu notebook): 3487 · Thử thách cá nhân: **#2**

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

Bộ kiểm tra: **15/15 ✅** (TODO 1.1 → 4.4 + 5 phép tính tay). Đủ 10 dự đoán. Notebook chạy trên CPU (máy cá nhân, nbconvert).
Phần 4 gọi API thật (`LLM_PROVIDER = "openai"`, `gpt-6-luna` / `gpt-6.1-terra`). Tổng chi phí cả bài < $0.01.

## 2. Kết quả chính

| Phương án | precision | recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (lô) | p50 ms | p95 ms | USD/1k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.986 | 0.802 | 0.885 | 0.947 | 2.1 | 31 520 | 0.07 | **0.09** | 0 |
| 2. BoW + MLP | 0.988 | 0.942 | **0.964** | 1.000 | 4.1 | 36 828 | 0.26 | **0.31** | 0 |
| 3a. Encoder + kNN | 0.888 | 0.919 | 0.903 | 0.976 | 26.2 | 39 | 41.0 | 44.1 | 0 |
| 3b. Encoder + LogReg | 0.920 | 0.942 | 0.931 | 0.976 | 26.6 | 103 | 22.9 | 25.4 | 0 |
| 3c. Encoder + MLP head | 0.930 | 0.930 | 0.930 | 1.000 | 27.8 | 102 | 23.5 | 27.3 | 0 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`) | 1.000 | 0.950 | — | 0.974 | 0 | 0.74 | 1 084 | 2 705 | 0.007 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) | 1.000 | 1.000 | — | **1.000** | 0 | 0.61 | 1 301 | 3 356 | 0.070 |
| 4c. LLM nhỏ few-shot | 1.000 | 0.950 | — | 0.974 | 0 | 0.56 | 1 036 | 5 446 | 0.023 |

Hàng LLM chỉ đo trên SUB 40 tin (precision/recall của hàng 4 là trên SUB).
Đường Pareto (SUB): BoW + LogReg → BoW + MLP. Đạt p95 < 50 ms: hàng 1, 2, 3a, 3b, 3c. LLM chậm hơn ~100 lần so với SLA.

## 3. Dự đoán và thực tế

1. **`hard_eval`: "LLM ~75–85% trên 50 case khó".** Thực tế **100% (50/50)**, cả prompt thường (4a) lẫn prompt phòng thủ (4d).
   Các model tự train chỉ còn 54–72%. Tôi đánh giá thấp khoảng cách giữa "học khuôn từ 3.600 tin" và "hiểu ý định người gửi".
2. **`dl_curve`: "MLP ≈ LogReg (±0.02)".** Thực tế MLP 0.964 so với LogReg 0.885. Nhưng bản *không ReLU* đạt 0.959, nên lợi thế
   không đến từ hidden layer mà từ cách train (Adam + mini-batch + 30 epoch, so với 300 bước GD còn underfit: recall train chỉ 0.847).
3. **`ml_lr`: "lr=50 dao động, không hội tụ mượt".** Loss vọt lên ~2.6 ở vài bước đầu rồi về **0.005**, thấp hơn cả lr=0.5 (0.083).
   Dữ liệu BoW gần tách tuyến tính, nên bước lớn chỉ làm ‖w‖ tăng nhanh. Loss train thấp nhưng mô hình quá tự tin.

Dự đoán đúng: model lớn hơn model nhỏ ~0.03 F1 (1 tin) và đắt **10 lần**. Few-shot không đổi F1 mà tốn **~3.3 lần**. LLM không thắng rõ 3b trên SUB.

## 4. Phân tích 50 case khó

- Model tự train tụt từ 0.95–1.00 (SUB) xuống **0.38–0.74**. **BoW + MLP tụt nhiều nhất** (1.000 → 0.378, acc 54%). **Encoder + kNN** bền nhất
  trong nhóm tự train (0.741, acc 72%). **LLM không tụt** (F1 1.000, đúng 100% ở cả 7 nhóm).
- BoW gần như mù với *Né từ khoá* (MLP 0%, LogReg 17%), *Spam giọng tự nhiên* (10–20%) và *Thao túng* (12–25%), nhưng đạt 100% ở *Nói về spam*.
  Embedding ngược lại: bắt được *Né từ khoá* / *Thao túng* (kNN 83–88%) nhưng chỉ 50% ở *Nói về spam*.
- **Case #31** `U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061…`: **mọi model tự train** đều sai. Với BoW, mỗi chữ cái thành
  một token ngoài vocab hoặc vô nghĩa. Với encoder, chuỗi ký tự rời làm vỡ sub-word nên vector không còn mang nghĩa "cash reward". LLM đọc xuyên được.
- **Case #47** `lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for that?`: là ham, nhưng Encoder + MLP head
  đoán spam. Vector câu nằm gần chính tin lừa đảo "won £1000, call 0906" vì encoder đo **chủ đề**, không đo **ý định** (kể lại để chê).
  BoW đúng nhờ "lol", "I", "people". LLM đúng vì hiểu đây là chuyện kể.
- 7 case làm mọi model tự train sai: 5 *Spam giọng tự nhiên* (#12, 13, 14, 16, 19), #22 (giả ngân hàng) và #31. LLM đúng cả 7.
- **Prompt injection:** 4a đã đúng 6/6, nên bộ case không đo được lợi ích của prompt phòng thủ (4d cũng 6/6, không làm hỏng nhóm nào).
- Bonus: 6 case tự viết (`MY_CASES`). Model tự train chỉ đúng 2–3/6, LLM đúng 6/6. Encoder nhầm OTP "staff will NEVER call you" và tin kể về cuộc gọi lừa đảo thành spam.

## 5. Thử thách cá nhân

**Đề #2:** đổi vocab 100 / 2000 / 5000, train lại LogReg.

| max_size | F1 test | recall | % token test ngoài vocab | train (s) |
| --- | ---: | ---: | ---: | ---: |
| 100 | 0.842 | 0.744 | 52.1% | 0.3 |
| 2000 | 0.885 | 0.802 | 13.6% | 1.8 |
| 5000 | 0.885 | 0.802 | 9.1% | 4.3 |

100 từ chủ yếu là từ chức năng, mất các từ spam đặc trưng, nên recall tụt (underfit). Lên 5000 thì F1 test không đổi: 3000 từ thêm là từ hiếm,
trọng số gần như không học được trong 300 bước GD, chỉ tăng gấp đôi thời gian train. Lợi ích của vocab bão hoà quanh ~2000 từ.

## 6. Phương án cho sản phẩm thật

Yêu cầu p95 < 50 ms, 10 triệu SMS/ngày (~116 tin/s). Chọn **BoW + MLP** làm tầng chính: p95 0.31 ms, ~36 800 tin/s trên một CPU,
F1 test 0.964, precision 0.988 (ít chặn nhầm OTP), chi phí ≈ 0. Encoder (3b/3c, p95 25–27 ms) cũng đạt SLA nhưng F1 thấp hơn.
LLM không đạt SLA (p95 2.7–5.4 s). Nếu dùng cho mọi tin thì tốn **~$70/ngày ≈ $2 100/tháng** (`gpt-6-luna`, $0.007/1k tin)
hoặc **~$21 000/tháng** (`gpt-6.1-terra`).
**Đánh đổi chấp nhận:** BoW yếu với spam né từ khoá và spam giọng tự nhiên (stress test 54%), trong khi LLM đúng 100%. Giảm rủi ro bằng **router**:
BoW + MLP chặn hoặc cho qua các tin chắc chắn. Tin có p ∈ [0.3, 0.7] chuyển sang Encoder + kNN, và LLM nhỏ với prompt phòng thủ chỉ chạy
**bất đồng bộ** cho phần nhỏ còn lại (5% lưu lượng ≈ $105/tháng). Nhãn của LLM được dùng để train lại model rẻ định kỳ.
Thêm bước chuẩn hoá leetspeak / chữ tách rời và blacklist đầu số tính phí.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI làm gì | Kiểm chứng |
| --- | --- | --- | --- |
| 4 | Viết `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` theo TODO 4.1–4.4 (Claude Code, Opus 5.5) | Code gọi Chat Completions với `**provider_kwargs(max_tokens)`, prompt zero/few-shot, parse nhãn chịu lỗi, tính chi phí theo token ước tính | Đọc lại trước khi gọi API; ô kiểm tra bằng API giả lập ✅ |
| 4 | Lỗi `TypeError: process() takes no keyword arguments`; LLM F1 = 0 ở lần chạy đầu | Nâng `brotli` 1.0.9 → 1.2.1; phát hiện `max_completion_tokens=3` làm `gpt-6-luna` trả `content=''` (`finish_reason="length"`), đổi thành 16 | Gọi API thử với 3 và 50 token; chạy lại notebook: invalid = 0, F1 LLM 0.974–1.000 |

**Điều hiểu ra:** muốn biết một thành phần (hidden layer) có tác dụng không thì phải làm ablation. MLP thắng LogReg nhưng bản bỏ ReLU cũng thắng,
nên lý do thật là cách tối ưu. Điểm F1 ~0.9–1.0 trên test cùng phân phối gần như không nói gì về độ bền: các model tự train chỉ còn 0.38–0.74
trên 50 case khó, còn LLM giữ 1.000 nhưng chậm hơn ~10 000 lần và tốn tiền. Một lỗi cấu hình nhỏ (`max_tokens=3`) có thể làm F1 = 0 mà không báo
lỗi, nên phải kiểm tra cột `invalid` trước khi tin vào số liệu.
