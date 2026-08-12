# 10 câu hỏi phản biện về Metric — chuẩn bị bảo vệ đồ án

## Nhóm lý thuyết (metric là gì, đo cái gì)

**1. AUC-ROC là gì? Vì sao vẫn dùng được cho dữ liệu mất cân bằng (22%/78%)?**
Gợi ý trả lời: AUC-ROC = diện tích dưới đường cong ROC (True Positive Rate vs False Positive Rate qua mọi ngưỡng phân loại). Đo khả năng **phân biệt thứ hạng** giữa 2 lớp — không phụ thuộc ngưỡng cụ thể, không bị chi phối bởi tỷ lệ lớp (vì TPR/FPR đều là tỷ lệ *trong nội bộ từng lớp*, không phải trên toàn bộ dữ liệu) — đây là lý do AUC-ROC ít bị "lừa" bởi mất cân bằng hơn Accuracy.

**2. PR-AUC (Precision-Recall AUC) khác AUC-ROC ở điểm nào? Vì sao với dữ liệu mất cân bằng, PR-AUC được coi là chỉ số "khắt khe" hơn?**
Gợi ý: PR-AUC dùng Precision thay cho FPR trên trục — mà Precision = TP/(TP+FP) **phụ thuộc trực tiếp vào tỷ lệ lớp dương** trong dữ liệu. Khi lớp dương hiếm (22%), số False Positive tuyệt đối dễ áp đảo, kéo Precision xuống thấp ngay cả khi FPR (theo AUC-ROC) vẫn nhỏ — nên PR-AUC (trong báo cáo: 0,51–0,57) luôn thấp hơn nhiều so với AUC-ROC (0,75–0,79) cho cùng 1 model, đó là điều bình thường, không phải model kém.

**3. G-mean là gì? Vì sao dùng trung bình NHÂN (căn bậc 2 của tích) thay vì trung bình cộng của Recall và Specificity?**
Gợi ý: G-mean = √(Recall × Specificity). Trung bình nhân trừng phạt mạnh sự lệch pha giữa 2 phía — nếu 1 trong 2 gần 0 (vd. model chỉ đoán đúng lớp đa số), G-mean sụp xuống gần 0 dù trung bình cộng vẫn có thể ở mức trung bình (vd. Recall=0,1, Specificity=0,95 → trung bình cộng=0,525 nhưng G-mean=0,31). Đây là lý do G-mean "khó gian lận" hơn với dữ liệu mất cân bằng.

**4. Brier score đo gì? Khác biệt căn bản với AUC-ROC là gì?**
Gợi ý: Brier = trung bình[(xác suất dự đoán − nhãn thật)²] — đo độ chính xác của **con số xác suất tuyệt đối** (calibration). AUC-ROC chỉ đo **thứ hạng tương đối** (ai rủi ro hơn ai), hoàn toàn không quan tâm bản thân con số xác suất có đúng hay không — 1 model có thể có AUC-ROC hoàn hảo (1.0) nhưng Brier score tệ nếu xác suất dự đoán bị lệch hệ thống (vd. luôn dự đoán 0,9 cho ca dương thay vì đúng phải là 0,6).

**5. Vì sao Accuracy bị coi là chỉ số gây hiểu lầm (misleading) trên bộ dữ liệu này? Cho ví dụ số cụ thể.**
Gợi ý: Với tỷ lệ vỡ nợ 22,12%, một model "ngây thơ" luôn dự đoán "không vỡ nợ" cho mọi khách hàng vẫn đạt Accuracy = 77,88% — cao hơn cả nhiều model thật trong báo cáo (~73–77%) — nhưng vô dụng hoàn toàn (Recall = 0%, không phát hiện được bất kỳ ai vỡ nợ). Đây là lý do Accuracy không được dùng làm tiêu chí chọn model chính.

## Nhóm lý do lựa chọn (vì sao làm như vậy)

**6. Vì sao chọn AUC-ROC làm tiêu chí chính để chọn best_params trong Grid Search/5-fold CV, thay vì Accuracy hay F1?**
Gợi ý: AUC-ROC không phụ thuộc ngưỡng phân loại (threshold-independent) — phù hợp ở giai đoạn *chọn tham số mô hình*, khi ngưỡng phân loại cuối cùng chưa được quyết định (và có thể tối ưu riêng sau, theo chi phí nghiệp vụ). F1/Accuracy phụ thuộc ngưỡng 0,5 mặc định — chọn tham số theo 1 chỉ số phụ thuộc ngưỡng chưa tối ưu có thể dẫn đến kết luận sai lệch nếu sau này đổi ngưỡng.

**7. Vì sao ở version 7 lại tính đủ 8 metric cho mỗi fold trong CV, thay vì chỉ AUC-ROC như bản trước?**
Gợi ý: Chỉ tính AUC-ROC thì không biết được bộ tham số "thắng" theo AUC có đánh đổi Precision/Recall/F1 kém hơn bộ khác trong cùng lưới hay không. Kết quả thực tế cho thấy với 5/6 model, chênh lệch không đáng kể — nhưng với Random Forest, chọn theo AUC bỏ lỡ tới 3,67 điểm % Recall so với mức tối đa có thể đạt trong cùng lưới — nếu không đo, sẽ không bao giờ phát hiện ra đánh đổi này.

**8. Ngưỡng phân loại 0,5 (dùng để tính Accuracy/Precision/Recall/F1) có phải luôn là lựa chọn tối ưu không? Vì sao bài toán ưu tiên thu hồi nợ cần cân nhắc đổi ngưỡng?**
Gợi ý: 0,5 là ngưỡng mặc định, không dựa trên chi phí nghiệp vụ thực tế. Với bài toán thu hồi nợ, bỏ sót 1 khách hàng vỡ nợ thật (False Negative) thường tốn kém hơn nhiều so với liên hệ nhầm 1 khách hàng tốt (False Positive) — nên hạ ngưỡng xuống dưới 0,5 để tăng Recall (chấp nhận đánh đổi Precision thấp hơn) thường hợp lý hơn về mặt kinh tế, dù thoạt nhìn "model tệ hơn" theo Accuracy.

**9. Precision và Recall đánh đổi nhau (trade-off) như thế nào? Trong bài toán ưu tiên thu hồi nợ nên ưu tiên chỉ số nào hơn?**
Gợi ý: Hạ ngưỡng phân loại → Recall tăng (bắt được nhiều ca dương hơn) nhưng Precision giảm (nhiều báo động giả hơn), và ngược lại — vì cùng 1 tập điểm xác suất, hạ ngưỡng đồng nghĩa gắn nhãn "dương" cho nhiều điểm hơn. Với thu hồi nợ, Recall thường quan trọng hơn (ưu tiên không bỏ sót khách rủi ro), miễn Precision không quá thấp đến mức làm quá tải nguồn lực bộ phận thu hồi nợ — đây là bài toán tối ưu có ràng buộc (constrained optimization), không phải "càng cao càng tốt" tuyệt đối cho 1 chỉ số.

**10. Vì sao không dùng thẳng Brier score hoặc Accuracy làm tiêu chí chọn tham số tốt nhất trong CV, mà lại chọn AUC-ROC?**
Gợi ý: Accuracy bị lệch bởi tỷ lệ lớp (câu 5). Brier score đo calibration (độ chính xác xác suất tuyệt đối) — quan trọng cho `priority_score`, nhưng một model có thể có Brier tốt (do dự đoán "an toàn" gần tỷ lệ nền cho mọi người) mà vẫn phân biệt kém giữa các khách hàng cụ thể (thứ hạng rủi ro sai) — không phù hợp làm tiêu chí *chọn giữa các bộ tham số* cho mục tiêu phân loại. AUC-ROC cân bằng tốt giữa 2 nhu cầu (không lệch theo tỷ lệ lớp + đo đúng khả năng phân biệt) nên phù hợp nhất làm tiêu chí chọn tham số, dù không hoàn hảo (đã thảo luận ở câu 4).
