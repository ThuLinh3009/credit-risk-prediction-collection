# Kịch bản thuyết trình — Hệ thống Dự đoán Rủi ro Tín dụng và Đề xuất Thu hồi nợ

> Ghi chú: kịch bản viết theo 4 slide (Title → Tổng quan bài toán → Dataset & Flow →
> Kết quả & So sánh paper gốc). Nếu bạn muốn giữ slide 4 là "Tech Stack" thay vì
> "Kết quả", chỉ cần bỏ qua phần narration Slide 4 bên dưới và nói ngắn gọn về công cụ
> sử dụng thay thế. Thời lượng ước tính: ~6-7 phút nói + phần hỏi đáp.

---

## Mở đầu (~20 giây)

> "Chào thầy/cô và các bạn, em là Đoàn Thị Thu Linh. Đề tài của em là **Hệ thống Dự
> đoán Rủi ro Tín dụng và Đề xuất Thu hồi nợ**, xây dựng trên bộ dữ liệu công khai UCI
> Default of Credit Card Clients, đối chiếu với nghiên cứu gốc của Yeh & Lien năm 2009."

*(chuyển slide 1 → slide 2)*

---

## Slide 2 — Tổng quan bài toán (~1.5 phút)

> "Trước khi vào chi tiết, em muốn làm rõ 3 điều: bài toán này là gì, vì sao em chọn
> đề tài này, và cụ thể em giải quyết cái gì.
>
> **Thứ nhất, bài toán là gì.** Nói đơn giản: em dùng lịch sử trả nợ 6 tháng của
> 30.000 khách hàng thẻ tín dụng để dự đoán trước ai sắp không trả nổi nợ tháng tới,
> sau đó đề xuất xem trong số đó nên ưu tiên gọi điện đòi nợ ai trước — giống cách một
> phòng cấp cứu phân loại bệnh nhân theo mức độ nguy hiểm, chứ không xử lý dàn trải.
>
> **Thứ hai, vì sao em chọn đề tài này.** Vỡ nợ thẻ tín dụng là vấn đề rất thực tế của
> ngành ngân hàng, đặc biệt ở các thị trường mới nổi. Bộ dữ liệu UCI này đặc biệt ở chỗ
> có sẵn một nghiên cứu gốc từ năm 2009 để em đối chiếu — cho phép em vừa làm được phần
> học thuật là so sánh mô hình hiện đại với 15 năm trước, vừa làm được phần ứng dụng là
> xây một công cụ hỗ trợ thu hồi nợ thực tế.
>
> **Thứ ba, cụ thể em giải quyết 2 lớp bài toán khác bản chất.** Lớp thứ nhất là
> *predictive* — bài toán phân loại nhị phân có huấn luyện, dùng 4 mô hình Logistic
> Regression, Random Forest, XGBoost và Neural Network để dự đoán xác suất vỡ nợ, gọi
> là `risk_score`. Lớp thứ hai là *prescriptive* — một công thức xếp hạng dựa trên luật,
> `priority_score = risk_score nhân với mức độ trễ hạn`, để xác định khách hàng nào cần
> gọi điện thu hồi nợ trước. Em cố tình tách 2 lớp này vì phần ưu tiên không có nhãn dữ
> liệu để huấn luyện — không ai ghi lại 'lẽ ra nên gọi ai trước' — nên bắt buộc phải
> dùng luật do con người thiết kế, chứ không thể học từ dữ liệu."

*(chuyển slide 2 → slide 3)*

---

## Slide 3 — Dataset & Quy trình xử lý (~2 phút)

> "Về dữ liệu: UCI Default of Credit Card Clients có 30.000 khách hàng thẻ tín dụng
> tại Đài Loan, 23 biến đầu vào và 1 biến mục tiêu nhị phân là có vỡ nợ tháng sau hay
> không. Tỷ lệ vỡ nợ khoảng 22%, tức dữ liệu khá mất cân bằng — điều này ảnh hưởng
> trực tiếp đến cách em chọn metric đánh giá, em sẽ nói ở slide sau.
>
> 23 biến này chia làm 4 nhóm: nhân khẩu học như hạn mức, tuổi, học vấn; lịch sử trả
> nợ 6 tháng gần nhất; dư nợ hóa đơn hàng tháng; và số tiền đã thanh toán mỗi tháng.
>
> Quy trình xử lý em chia làm 7 bước, đi từ trái qua phải rồi xuống hàng dưới. Bắt đầu
> từ làm sạch dữ liệu — ví dụ cột `EDUCATION` có vài giá trị không nằm trong định nghĩa
> gốc, em gộp lại thành nhóm 'khác'. Tiếp theo là feature engineering — vì 6 cột dư nợ
> hàng tháng tương quan rất cao với nhau, em tổng hợp chúng thành vài đặc trưng đại
> diện như dư nợ trung bình và mức độ trễ hạn tệ nhất trong 6 tháng, để giảm nhiễu cho
> mô hình.
>
> Sau đó em chia tập train/validation/test theo tỷ lệ 60/20/20, huấn luyện và tinh
> chỉnh cả 4 mô hình, đánh giá và so sánh với paper gốc, dùng SHAP để diễn giải mô
> hình xem yếu tố nào ảnh hưởng mạnh nhất, và cuối cùng tính `priority_score` để ra
> bảng xếp hạng ưu tiên thu hồi nợ."

*(chuyển slide 3 → slide 4)*

---

## Slide 4 — Kết quả & So sánh với paper gốc (~1.5-2 phút)

> "Về kết quả: trong 4 mô hình so sánh, **XGBoost cho hiệu suất tốt nhất** với AUC-ROC
> khoảng 0.79 trên tập test.
>
> AUC-ROC ở đây có thể hiểu đơn giản là: nếu bốc ngẫu nhiên một khách hàng vỡ nợ và một
> khách hàng không vỡ nợ, có khoảng 79% khả năng mô hình chấm điểm rủi ro của người vỡ
> nợ cao hơn. Em chọn AUC-ROC làm chỉ số chính thay vì Accuracy, vì dữ liệu mất cân
> bằng 22% khiến Accuracy dễ gây hiểu lầm — một mô hình chỉ đoán 'không ai vỡ nợ cả'
> vẫn đạt Accuracy 78% dù hoàn toàn vô dụng.
>
> Khi đối chiếu với paper gốc Yeh & Lien 2009 — quy đổi từ chỉ số area ratio trong lift
> chart mà paper gốc sử dụng sang AUC tương đương — thì Logistic Regression của em đạt
> khoảng 0.74, cải thiện so với khoảng 0.72 của paper gốc; còn Neural Network đạt
> khoảng 0.78, cải thiện so với khoảng 0.77 của paper gốc. Em xin nhấn mạnh đây là con
> số quy đổi tương đối, không phải AUC-ROC do tác giả gốc trực tiếp công bố, vì phương
> pháp đánh giá hai bên khác nhau.
>
> Về diễn giải mô hình, phân tích SHAP cho thấy 3 yếu tố ảnh hưởng mạnh nhất đến dự
> đoán vỡ nợ là: mức độ trễ hạn tệ nhất trong 6 tháng, trạng thái trễ hạn ở tháng gần
> nhất, và hạn mức tín dụng — đúng với trực giác nghiệp vụ, không phải mô hình đang dựa
> vào yếu tố không liên quan."

*(chuyển slide 4 → kết luận)*

---

## Kết luận (~30 giây)

> "Tóm lại, em đã xây dựng được một hệ thống 2 lớp: lớp dự đoán rủi ro vỡ nợ bằng 4 mô
> hình machine learning, được diễn giải rõ ràng bằng SHAP và đối chiếu được với nghiên
> cứu gốc 2009; và lớp đề xuất thu hồi nợ bằng công thức xếp hạng ưu tiên, giúp bộ phận
> thu hồi nợ tập trung nguồn lực đúng chỗ thay vì liên hệ dàn trải. Em xin dừng phần
> trình bày ở đây và sẵn sàng nhận câu hỏi từ thầy/cô."

---

## Phụ lục — Câu hỏi phản biện thường gặp & cách trả lời ngắn gọn

**Q: Vì sao phải làm feature engineering, không dùng thẳng dữ liệu gốc?**
> Vì 6 cột dư nợ hàng tháng tương quan rất cao với nhau (đa cộng tuyến), đưa thẳng vào
> Logistic Regression sẽ gây nhiễu. Em tổng hợp thành các đặc trưng đại diện như dư nợ
> trung bình và mức trễ hạn tệ nhất — và thực tế, đặc trưng tự tạo `MAX_DELAY` lại là
> yếu tố ảnh hưởng mạnh nhất theo SHAP.

**Q: SHAP là gì?**
> Là công cụ giải thích model "hộp đen" — cho biết mỗi đặc trưng đóng góp bao nhiêu vào
> con số dự đoán cuối cùng, giống như tách điểm thi tổng thành điểm từng môn.

**Q: Vì sao không dùng Accuracy để đánh giá?**
> Vì dữ liệu mất cân bằng 22% vỡ nợ khiến Accuracy dễ đánh lừa — một model luôn đoán
> "không vỡ nợ" vẫn đạt ~78% Accuracy mà vô dụng. Em dùng AUC-ROC làm chỉ số chính.

**Q: Kết hợp bài toán dự đoán (ML) và bài toán đề xuất (rule-based) trong 1 hệ thống
có hợp lý không?**
> Có — đây là kiến trúc chuẩn của ngành (predictive feed vào prescriptive). Không thể
> gộp thành 1 model ML vì không có nhãn "nên ưu tiên gọi ai trước" để huấn luyện.

**Q: Công thức `priority_score = risk_score × delay_severity` có được kiểm định thống
kê không?**
> Đây là công thức rule-based do em thiết kế dựa trên logic nghiệp vụ, không huấn
> luyện từ dữ liệu nên không có AUC/Accuracy để đánh giá nó. Em kiểm chứng tính hợp lý
> qua case study định tính — so sánh khách hàng rủi ro cao đã trễ hạn với khách hàng
> rủi ro cao nhưng chưa trễ hạn, để đảm bảo thứ hạng đầu ra đúng ý đồ nghiệp vụ. Đây
> cũng là hạn chế em đã tự nêu trong đề xuất dự án.

**Q: Sao không dùng Optuna như trong proposal ban đầu?**
> Em dùng grid search thủ công với lưới tham số nhỏ, tương đương phương án thay thế
> GridSearchCV mà proposal cũng đã đề cập, để đảm bảo thời gian chạy trong phạm vi cho
> phép.
