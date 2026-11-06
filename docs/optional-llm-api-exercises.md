# Ngày 1 — Bài Tập & Phản Ánh (side lab TUỲ CHỌN)

> Side lab này **không** thuộc điểm chính của Day 01 (xem [RUBRIC.md](../RUBRIC.md) mục Bonus). Model: `gpt-6.1-terra` (lớn) và `gpt-6-luna` (nhỏ). `gpt-4o` / `gpt-4o-mini` bị gỡ khỏi API ngày 14/01/2027 ([nguồn](https://developers.openai.com/api/docs/deprecations)). Tên khoá `gpt4o_*` / `mini_*` trong kết quả là hợp đồng của test, giữ nguyên để tương thích: đọc là "model lớn" / "model nhỏ".
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Copy `src/optional_llm_api/template.py` thành `src/optional_llm_api/solution/solution.py`, triển khai mọi TODO, rồi chạy từ gốc repo: `pytest tests/test_solution.py -v`. Chưa có `solution/solution.py` thì test tự bỏ qua.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python src/optional_llm_api/solution/solution.py
```
Bạn sẽ thấy output so sánh phản hồi của `gpt-6.1-terra` và `gpt-6-luna`.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."** (nhớ gửi `reasoning_effort="none"`: model GPT-6 chỉ nhận `temperature` khi tắt suy luận)

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Chạy `call_openai` (`gpt-6.1-terra`, `top_p=0.9`, `max_completion_tokens=150`, `reasoning_effort="none"`), mỗi mức temperature gọi 2 lần:
>
> | T | Lần 1 | Lần 2 |
> |---|---|---|
> | 0.0 | cà phê trứng / xuất khẩu cà phê | hang Sơn Đoòng |
> | 0.5 | Sơn Đoòng | Sơn Đoòng ("có cả những đám mây riêng!") |
> | 1.0 | Sơn Đoòng | Việt Nam xuất khẩu hạt điều lớn nhất thế giới |
> | 1.5 | cà phê trứng + robusta | Sơn Đoòng |
>
> Quy luật: temperature càng cao thì **chủ đề và cách diễn đạt càng đa dạng** (ở T = 1.0 mới xuất hiện chủ đề mới "hạt điều", T thấp chủ yếu xoay quanh Sơn Đoòng). Nhưng không phản hồi nào bị "loạn chữ" kể cả ở T = 1.5, vì `top_p=0.9` đã cắt bỏ phần đuôi token xác suất thấp. Bất ngờ: **T = 0.0 vẫn cho 2 câu trả lời khác nhau** (cà phê trứng so với Sơn Đoòng). API không đảm bảo xác định tuyệt đối, do batching và phần cứng phía server, nên không thể dựa vào T = 0 để có output lặp lại 100%.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Khoảng **0.2–0.3** (hoặc 0 nếu là câu trả lời theo quy trình như chính sách đổi trả, tra cứu đơn hàng). Chatbot hỗ trợ khách hàng cần **nhất quán và chính xác**: cùng câu hỏi thì cùng thông tin, không "sáng tạo" ra chính sách hay con số. Temperature thấp giảm rủi ro bịa đặt, nhưng một chút ngẫu nhiên giúp câu chữ tự nhiên hơn, không lặp máy móc. Như thí nghiệm trên, kể cả T = 0 vẫn có thể khác nhau, nên các thông tin quan trọng phải lấy từ dữ liệu thật (RAG, tra cứu hệ thống) chứ không dựa vào temperature.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem `gpt-6.1-terra` đắt hơn `gpt-6-luna` bao nhiêu lần cho workload này** (giá 11/2026: terra $1.00 / $5.00, luna $0.10 / $0.50 mỗi 1M token input / output):
> Workload: 10.000 người × 3 lần × ~350 token = **10,5 triệu token/ngày**. Giá terra gấp **đúng 10 lần** luna ở cả input ($1.00 so với $0.10) lẫn output ($5.00 so với $0.50), nên dù tỉ lệ input/output thế nào thì **terra luôn đắt hơn 10 lần**.
>
> Ví dụ mỗi lần gọi 250 token input + 100 token output: input 7,5M token/ngày, output 3M token/ngày.
> - `gpt-6-luna`: 7,5 × $0.10 + 3 × $0.50 = **$2.25/ngày ≈ $67.5/tháng**
> - `gpt-6.1-terra`: 7,5 × $1.00 + 3 × $5.00 = **$22.5/ngày ≈ $675/tháng**
>
> Nếu toàn bộ 350 token là output (trường hợp đắt nhất): luna $5.25/ngày, terra $52.5/ngày, vẫn đúng 10 lần.

**Mô tả một trường hợp mà chi phí cao hơn của `gpt-6.1-terra` là xứng đáng, và một trường hợp `gpt-6-luna` là lựa chọn tốt hơn:**
> **Terra xứng đáng** khi một câu trả lời sai rất đắt và số lượng gọi ít: ví dụ trợ lý phân tích hợp đồng hay tư vấn tài chính, suy luận nhiều bước, viết code phức tạp. Ở đây chênh vài cent mỗi lần gọi không đáng kể so với chi phí của một lỗi. **Luna tốt hơn** cho tác vụ đơn giản, khối lượng lớn, có thể kiểm chứng: phân loại spam, gắn nhãn, trích xuất trường, FAQ chatbot. Trong lab lọc spam, luna đạt F1 0.974 trên SUB so với 1.000 của terra (chỉ chênh 1 tin trên 40) nhưng rẻ hơn 10 lần ($0.007 so với $0.070 mỗi 1.000 tin). Cách lai: dùng luna mặc định, chỉ chuyển sang terra khi luna không chắc chắn (router).

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi **có người đang chờ đọc câu trả lời dài** (chatbot, trợ lý viết, giải thích): tổng thời gian không đổi, nhưng **thời gian đến token đầu tiên** giảm mạnh, chữ hiện dần nên người dùng thấy hệ thống phản hồi ngay và có thể đọc hoặc dừng sớm. Điều này càng quan trọng khi độ trễ API cao như lab này (5–30 s cho một lần gọi). Non-streaming phù hợp hơn khi **output được máy xử lý chứ không phải người đọc**: phân loại một nhãn (spam/ham), JSON cần parse trọn vẹn, xử lý batch hay pipeline nền, hoặc khi cần kiểm duyệt toàn bộ nội dung trước khi hiển thị. Lúc đó streaming chỉ thêm phức tạp (ghép chunk, xử lý lỗi giữa chừng) mà không mang lại lợi ích gì.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/test_solution.py -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
