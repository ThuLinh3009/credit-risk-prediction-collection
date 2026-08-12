# Kịch bản thuyết trình chi tiết — Version 7 (bao_cao_ket_qua_v7.pptx)

> Kịch bản theo đúng 14 slide của `bao_cao_ket_qua_v7.pptx`. Đi sâu vào **cách làm**,
> **vì sao làm như vậy**, và **cơ chế hoạt động của các khái niệm kỹ thuật** (Grid
> Search + 5-fold CV, AUC-ROC...) — phù hợp đọc để thuyết trình hoặc trả lời phản biện.
> Thời lượng ước tính: ~14-16 phút nói (chưa tính hỏi đáp). Có thể cắt bớt phần "đi sâu
> cơ chế" nếu thời gian thuyết trình bị giới hạn — đã đánh dấu rõ đoạn nào có thể lược.

---

## Slide 1 — Trang bìa (~20 giây)

> "Chào thầy/cô và các bạn. Em là Đoàn Thị Thu Linh, đề tài của em là **Hệ thống Dự
> đoán Rủi ro Tín dụng và Đề xuất Ưu tiên Thu hồi nợ**, xây dựng trên bộ dữ liệu công
> khai UCI Default of Credit Card Clients, đối chiếu với nghiên cứu gốc của Yeh & Lien
> năm 2009. Bản báo cáo hôm nay là phiên bản pipeline chính thức — em gọi là Version 7
> — dùng 6 mô hình, tinh chỉnh bằng Grid Search kết hợp 5-fold Cross-Validation, và xử
> lý mất cân bằng dữ liệu bằng kỹ thuật class-weight."

*(chuyển sang slide 2)*

---

## Slide 2 — Bối cảnh và Input/Output (~1.5 phút)

> "Trước khi vào chi tiết kỹ thuật, em làm rõ bài toán đang giải quyết cái gì.
>
> **Input** — đầu vào — là 23 biến của 30.000 khách hàng thẻ tín dụng, chia làm 4
> nhóm: nhân khẩu học như hạn mức tín dụng, tuổi, học vấn, hôn nhân; lịch sử trả nợ 6
> tháng gần nhất; dư nợ hóa đơn hàng tháng; và số tiền đã thanh toán mỗi tháng. Dữ liệu
> này khá sạch — em kiểm tra kỹ ở bước sau và xác nhận không có giá trị thiếu, không có
> dòng trùng lặp.
>
> **Output** — đầu ra — em thiết kế thành 3 tầng, không chỉ 1 con số duy nhất.
> Tầng thứ nhất là nhãn nhị phân — có vỡ nợ tháng sau hay không. Tầng thứ hai là
> `risk_score` — một xác suất liên tục từ 0 đến 1, do mô hình dự đoán. Tầng thứ ba là
> `priority_score` — kết hợp `risk_score` với mức độ trễ hạn thực tế của khách hàng, để
> ra một con số dùng trực tiếp cho nghiệp vụ: nên gọi điện đòi nợ ai trước.
>
> Em tách 3 tầng này vì mỗi tầng phục vụ một mục đích khác nhau: tầng 1 để huấn luyện
> mô hình, tầng 2 để có một con số liên tục thể hiện mức độ rủi ro thay vì chỉ đúng/sai,
> và tầng 3 mới là thứ bộ phận thu hồi nợ thực sự cầm trên tay để hành động."

*(chuyển sang slide 3)*

---

## Slide 3 — Quy trình 8 bước (~1 phút)

> "Toàn bộ quy trình em làm theo 8 bước, gần giống mô hình CRISP-DM chuẩn trong khoa
> học dữ liệu. Bắt đầu từ khám phá dữ liệu, phân tích tương quan, tiền xử lý — bao gồm
> làm sạch và tạo thêm đặc trưng mới — rồi đến 2 bước quan trọng nhất là huấn luyện
> baseline và tinh chỉnh tham số bằng Grid Search kết hợp Cross-Validation, sau đó đánh
> giá trên tập test, diễn giải mô hình bằng SHAP, và cuối cùng là xây công thức xếp
> hạng ưu tiên thu hồi nợ.
>
> Em sẽ đi qua từng bước, nhưng dành thời gian nhiều nhất cho bước 5 — vì đây là bước
> có nhiều quyết định kỹ thuật cần giải thích rõ vì sao em làm như vậy."

*(chuyển sang slide 4)*

---

## Slide 4 — EDA: Dữ liệu mất cân bằng (~1.5 phút)

> "Đây là điểm mấu chốt ảnh hưởng đến toàn bộ các quyết định kỹ thuật sau này: chỉ
> 22,12% khách hàng trong dữ liệu thực sự vỡ nợ, còn lại 77,88% không vỡ nợ.
>
> Vì sao con số này quan trọng đến vậy? Vì nó phá vỡ cách đánh giá mô hình thông
> thường. Thử tưởng tượng một mô hình 'lười biếng' nhất có thể — không học gì cả, cứ
> đoán đại mọi khách hàng đều 'không vỡ nợ'. Mô hình này vẫn đạt Accuracy — độ chính
> xác — tới 77,88%, cao hơn cả một vài mô hình thật sự có huấn luyện trong báo cáo của
> em. Nhưng nó hoàn toàn vô dụng, vì nó không phát hiện được bất kỳ ai vỡ nợ cả — Recall
> của nó bằng 0%.
>
> Đây chính là lý do em không dùng Accuracy làm thước đo chính xuyên suốt cả dự án — mà
> dùng AUC-ROC, và một loạt metric khác dành riêng cho dữ liệu mất cân bằng, em sẽ giải
> thích kỹ ở slide 7 và 9."

*(chuyển sang slide 5)*

---

## Slide 5 — Đa cộng tuyến và Feature Engineering (~1.5 phút)

> "Vấn đề dữ liệu thứ hai em phát hiện: 6 cột dư nợ hóa đơn hàng tháng — BILL_AMT1 đến
> BILL_AMT6 — tương quan với nhau cực kỳ cao. Em đo bằng hệ số VIF — Variance Inflation
> Factor — và cả 6 cột đều có VIF từ 13,8 đến 23,7, trong khi ngưỡng cảnh báo thông
> thường chỉ là 10. Điều này dễ hiểu về mặt nghiệp vụ: dư nợ hóa đơn của một khách hàng
> ít khi thay đổi đột ngột từ tháng này sang tháng khác, nên 6 cột gần như lặp lại cùng
> một thông tin.
>
> Đa cộng tuyến gây hại đặc biệt cho Logistic Regression — vì hệ số hồi quy sẽ trở nên
> bất ổn định khi các biến đầu vào tương quan quá cao với nhau.
>
> Giải pháp của em là tạo thêm 6 đặc trưng tổng hợp thay vì đưa nguyên 6 cột thô vào mô
> hình: dư nợ trung bình, xu hướng dư nợ đang tăng hay giảm, mức trễ hạn nặng nhất và
> trung bình trong 6 kỳ, và tỷ lệ trả nợ trên dư nợ. Và kết quả rất đáng chú ý — em sẽ
> nói ở slide 11 — chính đặc trưng tự tạo `MAX_DELAY` này lại là yếu tố ảnh hưởng mạnh
> nhất đến dự đoán, quan trọng hơn cả các cột dữ liệu gốc."

*(chuyển sang slide 6)*

---

## Slide 6 — Xử lý mất cân bằng: 6 mô hình, 6 cơ chế (~2 phút)

> "Quay lại vấn đề mất cân bằng ở slide 4. Em xử lý bằng kỹ thuật gọi là 'class
> weighting' — tăng trọng số ảnh hưởng của các ca vỡ nợ trong hàm mất mát (loss
> function) khi huấn luyện, để mô hình buộc phải chú ý đến nhóm thiểu số thay vì bỏ qua
> vì nó hiếm.
>
> Điểm đặc biệt là mỗi thư viện có API xử lý khác nhau, nên em phải áp dụng riêng cho
> từng mô hình: Logistic Regression, Random Forest và LightGBM có sẵn tham số
> `class_weight="balanced"` ngay trong hàm khởi tạo. XGBoost thì không có tham số này,
> nhưng có tham số tương đương là `scale_pos_weight` — tính bằng tỷ lệ số lượng lớp
> không-vỡ-nợ chia cho lớp vỡ-nợ, ra khoảng 3,52. Mô hình GBM của scikit-learn thì đặc
> biệt hơn — hoàn toàn không có tham số nào trong constructor cả, em phải tính
> `sample_weight` thủ công rồi truyền trực tiếp vào lúc gọi hàm `fit()`.
>
> Riêng Neural Network — Multi-Layer Perceptron — là ngoại lệ duy nhất, vì thư viện
> hoàn toàn không hỗ trợ cả `class_weight` lẫn `sample_weight`. Em phải oversample thủ
> công — nhân bản ngẫu nhiên các ca vỡ nợ trong tập train cho đến khi số lượng bằng với
> nhóm không vỡ nợ.
>
> Một quyết định quan trọng khác: em **không dùng SMOTE** — kỹ thuật sinh mẫu tổng hợp
> rất phổ biến để xử lý mất cân bằng — cho pipeline chính. Lý do: SMOTE tạo ra dữ liệu
> giả bằng nội suy giữa các điểm dữ liệu thật, làm cho mô hình học trên một phân phối
> lớp nhân tạo 50/50, khác hẳn tỷ lệ thật 22/78% — hệ quả là xác suất dự đoán ra bị lệch
> khỏi xác suất thật. Mà `risk_score` của em cần là một xác suất thật để nhân với mức độ
> trễ hạn ra `priority_score` — nên em ưu tiên giữ đúng phân phối gốc bằng class
> weighting, dù kỹ thuật này có phần thủ công hơn."

*(chuyển sang slide 7 — đây là slide quan trọng nhất, nên dành nhiều thời gian nhất)*

---

## Slide 7 — Grid Search + 5-fold Cross-Validation: cách hoạt động (~3 phút)

> "Đây là phần kỹ thuật trọng tâm của toàn bộ dự án, em xin giải thích kỹ.
>
> **Vấn đề cần giải quyết:** mỗi mô hình machine learning có những 'núm vặn' — gọi là
> siêu tham số (hyperparameter) — ảnh hưởng lớn đến hiệu năng. Ví dụ Random Forest cần
> biết trồng bao nhiêu cây, mỗi cây sâu tối đa bao nhiêu tầng. Chọn sai những con số
> này, mô hình có thể học kém hoặc học vẹt (overfitting). Vậy làm sao biết bộ tham số
> nào là tốt nhất?
>
> **Grid Search hoạt động như thế nào.** Cách đơn giản nhất — và cũng là cách em dùng —
> là liệt kê sẵn một danh sách nhỏ các giá trị hợp lý cho mỗi tham số, ghép chúng thành
> mọi tổ hợp có thể trong một 'lưới' — ví dụ XGBoost em thử 2 giá trị cho số lượng cây,
> 2 giá trị cho độ sâu, 2 giá trị cho tốc độ học, ghép lại ra 2 nhân 2 nhân 2 bằng 8 tổ
> hợp. Tổng cộng cho cả 6 mô hình là 38 tổ hợp tham số. Sau đó, thử LẦN LƯỢT từng tổ hợp,
> đo hiệu năng, và chọn tổ hợp cho kết quả tốt nhất.
>
> **Nhưng đo hiệu năng bằng cách nào mới đáng tin?** Đây mới là phần quan trọng. Cách
> đơn giản nhất là tách sẵn một phần dữ liệu validation cố định, train trên phần còn
> lại, đo trên phần validation đó. Nhưng cách này có một rủi ro: kết quả phụ thuộc vào
> đúng 1 lần chia dữ liệu cụ thể — nếu may mắn (hoặc xui) rơi vào 1 cách chia có nhiều ca
> vỡ nợ 'dễ đoán' hơn bình thường, kết luận 'tham số nào tốt nhất' có thể sai lệch mà
> không hề biết.
>
> **5-fold Cross-Validation giải quyết đúng vấn đề này.** Cách làm: chia toàn bộ 24.000
> dòng dữ liệu train thành 5 phần bằng nhau, gọi là 5 'fold'. Sau đó lặp lại 5 lần: mỗi
> lần, dùng 4 phần để huấn luyện, phần còn lại — chưa từng được nhìn thấy trong lần này
> — dùng để đánh giá. Làm sao để mỗi phần đều lần lượt được làm 'phần đánh giá' đúng 1
> lần, không trùng lặp, không bỏ sót. Sau 5 lần, ta có 5 con số hiệu năng độc lập — lấy
> trung bình để ra 1 con số đáng tin cậy hơn nhiều so với chỉ đánh giá 1 lần duy nhất.
> Chữ 'Stratified' nghĩa là khi chia 5 phần, em vẫn cố giữ đúng tỷ lệ 22% vỡ nợ trong
> mỗi phần — không để 1 fold nào tình cờ có tỷ lệ vỡ nợ quá lệch so với 4 fold còn lại.
>
> **Kết hợp cả hai:** với MỖI tổ hợp tham số trong lưới 38 tổ hợp, em chạy đủ 5-fold CV,
> lấy điểm AUC-ROC trung bình của 5 fold, rồi so sánh giữa các tổ hợp — tổ hợp nào có
> điểm trung bình cao nhất được chọn. Tổng cộng, riêng bước này em phải huấn luyện mô
> hình gần 200 lần — đây là lý do bước này chạy khá lâu, khoảng 24 phút cho cả 6 mô
> hình trên máy em.
>
> Một cải tiến em làm ở phiên bản này: thay vì mỗi fold chỉ tính đúng 1 con số AUC-ROC,
> em tính đủ 8 chỉ số cùng lúc — em sẽ trình bày kết quả này ở slide sau — để kiểm tra
> xem việc chọn tham số theo AUC-ROC có vô tình bỏ lỡ một tổ hợp khác cho Precision hay
> Recall tốt hơn hay không, thay vì chỉ giả định là không."

*(chuyển sang slide 8 — có thể lược ngắn phần trên nếu thời gian hạn chế, nhưng nên giữ
nguyên đoạn giải thích 5-fold CV vì đây thường là câu hỏi phản biện hay gặp nhất)*

---

## Slide 8 — Kết quả Cross-Validation đầy đủ metric (~1.5 phút)

> "Đây là bảng kết quả CV đầy đủ 8 chỉ số cho bộ tham số được chọn của mỗi mô hình. Em
> sẽ giải thích ý nghĩa từng chỉ số kỹ hơn ở slide sau, ở đây em chỉ nêu 2 phát hiện
> quan trọng nhất.
>
> Thứ nhất, khi em kiểm tra xem việc chọn tham số theo AUC-ROC có đánh đổi gì không —
> với 5 trong 6 mô hình, chênh lệch giữa 'chọn theo AUC-ROC' và 'tối ưu tối đa có thể
> đạt được theo F1 hoặc Recall' là rất nhỏ, dưới nửa điểm phần trăm — nghĩa là AUC-ROC
> và F1, Recall khá 'đồng thuận' với nhau trên bộ dữ liệu này.
>
> Nhưng với Random Forest thì khác — nếu chọn theo AUC-ROC, Recall chỉ đạt 60,07%,
> trong khi CÙNG lưới tham số đó, có một tổ hợp khác cho Recall lên tới 63,74% — chênh
> tới 3,67 điểm phần trăm. Nếu em không tính đủ 8 metric mà chỉ tính AUC-ROC như phiên
> bản trước, em sẽ không bao giờ phát hiện ra đánh đổi này. Đây là ví dụ cụ thể cho thấy
> vì sao việc đo đầy đủ nhiều chỉ số, thay vì chỉ 1 chỉ số duy nhất, mang lại giá trị
> thực sự — không chỉ là làm cho đẹp báo cáo."

*(chuyển sang slide 9)*

---

## Slide 9 — Kết quả tập Test và vì sao dùng AUC-ROC (~2.5 phút)

> "Sau khi chốt tham số ở bước CV, em huấn luyện lại mô hình trên toàn bộ 24.000 dòng
> train, rồi đánh giá đúng 1 lần duy nhất trên 6.000 dòng test — dữ liệu mà mô hình
> chưa từng nhìn thấy ở bất kỳ bước nào trước đó, kể cả lúc chọn tham số.
>
> **Vì sao em chọn AUC-ROC làm chỉ số chính để so sánh, xuyên suốt cả dự án?** Em giải
> thích theo 2 hướng: AUC-ROC là gì, và vì sao nó phù hợp hơn các lựa chọn khác.
>
> AUC-ROC có một cách hiểu rất trực quan: nếu em bốc ngẫu nhiên 1 khách hàng thực sự vỡ
> nợ và 1 khách hàng thực sự không vỡ nợ, AUC-ROC chính là xác suất mô hình chấm điểm
> rủi ro của người vỡ nợ CAO HƠN người không vỡ nợ. Với LightGBM đạt AUC-ROC 0,7886,
> nghĩa là khoảng 79% trường hợp mô hình xếp đúng thứ tự rủi ro giữa 2 người ngẫu nhiên
> như vậy.
>
> Vì sao không dùng Accuracy — như em đã nói ở slide 4, dữ liệu mất cân bằng làm
> Accuracy mất ý nghĩa. Còn vì sao không chỉ dùng Precision hay Recall đơn lẻ — vì cả
> 2 chỉ số này phụ thuộc vào một ngưỡng phân loại cụ thể, ở đây mặc định là 0,5, mà
> ngưỡng này chưa chắc là ngưỡng tối ưu về nghiệp vụ. AUC-ROC thì không phụ thuộc ngưỡng
> — nó đánh giá khả năng phân biệt của mô hình xuyên suốt MỌI ngưỡng có thể — nên phù
> hợp nhất để so sánh các mô hình trước khi quyết định ngưỡng cụ thể sẽ dùng.
>
> Về kết quả cụ thể: LightGBM, XGBoost và GBM — 3 mô hình boosting — cho AUC-ROC gần như
> bằng nhau, chênh nhau chưa tới 0,0004, nhỏ hơn nhiều so với độ dao động em quan sát
> được giữa các fold ở bước CV, khoảng 0,004. Về mặt thống kê, có thể coi 3 mô hình này
> tương đương nhau, LightGBM đứng 'hạng 1' chỉ mang tính hình thức. Random Forest và
> Neural Network thấp hơn một khoảng rõ rệt hơn, còn Logistic Regression tụt lại xa
> nhất — đúng như kỳ vọng cho một mô hình tuyến tính đơn giản trên một bài toán có quan
> hệ phi tuyến."

*(chuyển sang slide 10)*

---

## Slide 10 — Phân tích Confusion Matrix (~2 phút)

> "Con số AUC-ROC dù hữu ích để so sánh mô hình, nhưng khá trừu tượng về mặt nghiệp vụ.
> Em minh họa cụ thể hơn bằng confusion matrix của LightGBM — mô hình tốt nhất — ở
> ngưỡng phân loại mặc định 0,5.
>
> Trong 6.000 khách hàng tập test, có 1.327 người thực sự vỡ nợ. Mô hình phát hiện đúng
> 887 người — đây là True Positive, nhóm mô hình giúp bộ phận thu hồi nợ chủ động liên
> hệ sớm. Nhưng có tới 440 người vỡ nợ thật lại bị bỏ sót — False Negative — chiếm
> khoảng 33% tổng số ca vỡ nợ. Đây là rủi ro nghiệp vụ lớn nhất em muốn nhấn mạnh: cứ 3
> khách hàng sắp vỡ nợ thì mô hình bỏ lọt 1 người.
>
> Ở chiều ngược lại, có 1.021 khách hàng bị gắn nhầm là rủi ro cao dù thực tế vẫn trả nợ
> đúng hạn — False Positive — chi phí thấp hơn nhiều, chỉ là tốn công liên hệ không cần
> thiết. Còn 3.652 người được xác định đúng là an toàn.
>
> Điểm em muốn nhấn mạnh: ngưỡng 0,5 này là mặc định, chưa được tối ưu theo chi phí thực
> tế — mà bỏ sót 1 khách hàng vỡ nợ thường tốn kém hơn nhiều so với liên hệ nhầm 1 khách
> hàng tốt. Đây là một hướng cải thiện em đã ghi nhận nhưng chưa kịp triển khai trong
> phạm vi đồ án này."

*(chuyển sang slide 11)*

---

## Slide 11 — SHAP: Diễn giải mô hình (~1.5 phút)

> "Để trả lời câu hỏi 'mô hình dựa vào đâu để ra quyết định', em dùng SHAP — một kỹ
> thuật tính điểm đóng góp của từng đặc trưng vào dự đoán cuối cùng, giống như tách một
> điểm thi tổng thành điểm đóng góp của từng môn học.
>
> Kết quả: `MAX_DELAY` — đặc trưng em tự tạo ở bước feature engineering, đo mức trễ hạn
> nặng nhất trong 6 tháng — đứng đầu, quan trọng gấp đôi đặc trưng đứng thứ 2 là
> `LIMIT_BAL`, hạn mức tín dụng. Đứng thứ 3 là `PAY_1`, trạng thái trả nợ tháng gần
> nhất. Kết quả này khớp hoàn toàn với trực giác nghiệp vụ — khách hàng đã từng trễ hạn
> nặng và có hạn mức tín dụng thấp là nhóm rủi ro cao nhất — và cũng xác nhận việc tạo
> thêm đặc trưng `MAX_DELAY` ở bước 5 là một quyết định đúng đắn.
>
> Em cũng muốn nêu một điểm cần lưu ý: 2 biến nhân khẩu học — tình trạng hôn nhân và
> giới tính — vẫn lọt vào top 15 đặc trưng quan trọng. Đây là điều em xin phép nêu ra
> như một hạn chế cần đánh giá thêm, vì dùng đặc điểm nhân khẩu học trong quyết định tín
> dụng có thể gặp vấn đề về công bằng thuật toán ở một số khung pháp lý — em sẽ nói rõ
> hơn ở phần hạn chế cuối bài."

*(chuyển sang slide 12)*

---

## Slide 12 — Ứng dụng: priority_score (~2 phút)

> "Đây là sản phẩm ứng dụng cuối cùng của dự án — nơi `risk_score` được chuyển hóa
> thành một công cụ nghiệp vụ thực tế.
>
> Công thức: `priority_score = risk_score nhân với mức_độ_trễ_hạn`. `risk_score` là xác
> suất vỡ nợ mô hình dự đoán. Mức độ trễ hạn là giá trị lớn nhất trong 6 tháng gần nhất
> — chỉ tính số tháng trễ hạn thực tế, bỏ qua những trạng thái như trả đúng hạn hay
> đang dùng tín dụng xoay vòng.
>
> Kết quả thực tế trên tập test: nếu chia theo phân vị, nhóm 10% có `priority_score` cao
> nhất — 600 khách hàng — có `risk_score` trung bình 0,869 và từng trễ hạn trung bình
> 2,68 tháng. Bộ phận thu hồi nợ chỉ cần tập trung gọi điện trực tiếp cho 600 người này
> trước, thay vì liên hệ dàn trải toàn bộ 6.000 khách hàng — giảm khoảng 90% khối lượng
> công việc liên hệ trực tiếp ban đầu.
>
> Em cũng phát hiện một hạn chế khi phân tích kỹ: hơn 60% khách hàng trong tập test chưa
> từng trễ hạn — tức mức độ trễ hạn của họ bằng 0 — nên `priority_score` của họ cũng
> bằng 0, và con số 0 đó lại trùng đúng vào ngưỡng phân vị 60% em đặt ra ban đầu, khiến
> nhóm 'Thấp' bị rỗng hoàn toàn trên thiết kế 3 nhóm ban đầu. Đây là một hạn chế thực sự
> em phát hiện được khi phân tích kỹ số liệu, không phải giả định — và công thức cũng
> chưa tính đến giá trị dư nợ cụ thể của từng khách hàng, em sẽ nói rõ hơn ở slide kết
> luận."

*(chuyển sang slide 13)*

---

## Slide 13 — Kết luận và Hạn chế (~1.5 phút)

> "Tổng kết lại, dự án đạt được 4 điều chính: 3 mô hình boosting cho hiệu năng tương
> đương nhau và tốt nhất, quy trình Grid Search cộng 5-fold CV phát hiện được đánh đổi
> ẩn ở Random Forest mà cách làm cũ không thấy được, SHAP xác nhận đúng các yếu tố dự
> báo quan trọng nhất theo trực giác nghiệp vụ, và công thức ưu tiên giúp giảm 90% khối
> lượng công việc liên hệ trực tiếp.
>
> Về hạn chế, em xin nêu thẳng thắn 4 điểm: ngưỡng phân loại 0,5 khiến 33% ca vỡ nợ
> thật bị bỏ sót, chưa được tối ưu theo chi phí nghiệp vụ; công thức `priority_score`
> chưa tính đến giá trị dư nợ cụ thể; em chưa kiểm tra độ tin cậy của xác suất dự đoán —
> gọi là calibration — cho chính pipeline chính thức đang dùng; và biến nhân khẩu học
> vẫn nằm trong nhóm đặc trưng quan trọng, cần đánh giá thêm về công bằng thuật toán
> trước khi cân nhắc triển khai thực tế."

*(chuyển sang slide 14)*

---

## Slide 14 — Kết thúc (~15 giây)

> "Em xin dừng phần trình bày ở đây. Rất mong nhận được câu hỏi và góp ý từ thầy/cô và
> các bạn."

---

## Phụ lục A — Giải thích nhanh các metric hay bị hỏi lại (đọc khi cần, không thuộc kịch bản chính)

**PR-AUC là gì, khác AUC-ROC ở đâu?**
> "PR-AUC dùng Precision thay cho tỷ lệ báo động giả trên trục ngang. Vì Precision phụ
> thuộc trực tiếp vào tỷ lệ lớp dương trong dữ liệu, nên với dữ liệu mất cân bằng như
> của em, PR-AUC luôn thấp hơn nhiều so với AUC-ROC cho cùng 1 model — đây là điều bình
> thường, không phải model kém. Em xem PR-AUC là góc nhìn bổ sung, khắt khe hơn."

**G-mean là gì?**
> "G-mean là căn bậc 2 của tích Recall nhân Specificity — trung bình nhân, không phải
> trung bình cộng, của khả năng phát hiện đúng cả 2 lớp. Nếu model chỉ giỏi 1 phía,
> G-mean sẽ sụp xuống rất thấp — đây là lý do nó khó 'gian lận' hơn Accuracy trên dữ
> liệu mất cân bằng."

**Brier score là gì?**
> "Brier score đo sai số bình phương giữa xác suất dự đoán và nhãn thật — càng thấp
> càng tốt. Khác AUC-ROC ở chỗ: AUC-ROC chỉ quan tâm THỨ HẠNG tương đối, còn Brier quan
> tâm CON SỐ xác suất tuyệt đối có đúng hay không — quan trọng vì `risk_score` của em
> được dùng trực tiếp để tính `priority_score`."

**Vì sao không dùng Optuna như proposal ban đầu đề cập?**
> "Em dùng Grid Search thủ công với lưới tham số nhỏ — một trong 2 lựa chọn proposal đã
> nêu (Optuna hoặc GridSearchCV) — để đảm bảo thời gian chạy trong phạm vi cho phép,
> vì không gian tham số của em không quá lớn (38 tổ hợp cho 6 model), Grid Search đã đủ
> dò hết mà không cần thuật toán tìm kiếm thông minh hơn."

## Phụ lục B — Nếu bị hỏi "vì sao không dùng SMOTE ngay từ đầu, phải thử rồi mới bỏ?"

> "Em có thử SMOTE ở các phiên bản trước như một thí nghiệm so sánh riêng biệt, không
> trộn lẫn với pipeline chính. Kết quả kiểm tra bằng Brier score cho thấy: SMOTE khi
> chưa hiệu chỉnh lại làm xác suất dự đoán kém tin cậy hơn so với class weighting. Sau
> khi xác nhận điều này bằng số liệu thực nghiệm — không chỉ suy đoán lý thuyết — em
> quyết định loại SMOTE khỏi pipeline chính thức, chỉ giữ class weighting."

---

## Phụ lục C — Feature Engineering: cách làm chi tiết từng bước

> Đọc phần này nếu bị hỏi sâu ở slide 5, hoặc muốn tự tin hơn khi trình bày phần này.

**Thứ tự thực hiện — vì sao phải làm sạch trước rồi mới tạo đặc trưng:**
> "Em làm theo đúng thứ tự: (1) đổi tên `PAY_0` thành `PAY_1` để 6 cột trễ hạn có tên
> nhất quán `PAY_1` đến `PAY_6`, (2) gộp các giá trị lạ của `EDUCATION`/`MARRIAGE` vào
> nhóm 'khác', rồi (3) mới tính các đặc trưng tổng hợp. Thứ tự này bắt buộc — vì nếu
> tính `MAX_DELAY = max(PAY_1...PAY_6)` trước khi đổi tên `PAY_0`, hàm sẽ thiếu mất
> cột đầu tiên, tính sai ngay từ đầu."

**Từng đặc trưng — công thức và lý do:**

> "`AVG_BILL_AMT` và `AVG_PAY_AMT` — em lấy trung bình cộng của 6 cột dư nợ và 6 cột
> thanh toán. Ý tưởng đơn giản: thay vì đưa nguyên 6 con số gần như lặp lại nhau (nhớ
> lại VIF ở slide 5), em rút gọn thành 1 con số đại diện cho 'mức độ' chung, giảm nhiễu
> mà không mất nhiều thông tin.
>
> `BILL_AMT_TREND` — em lấy `BILL_AMT1` trừ `BILL_AMT6`, tức dư nợ tháng gần nhất trừ
> dư nợ tháng xa nhất trong 6 tháng. Đây là đặc trưng bổ sung THÔNG TIN MỚI mà trung
> bình cộng không có: hướng đi. Nếu dương, dư nợ đang tăng dần — dấu hiệu xấu đi. Nếu
> âm, dư nợ đang giảm dần — dấu hiệu cải thiện.
>
> `MAX_DELAY` và `AVG_DELAY` — tương tự, lấy giá trị lớn nhất và trung bình của 6 cột
> `PAY_1` đến `PAY_6`. `MAX_DELAY` nắm bắt tình huống XẤU NHẤT đã từng xảy ra trong 6
> tháng — và như em trình bày ở slide 11, đây lại chính là đặc trưng quan trọng nhất
> theo SHAP, quan trọng hơn cả `AVG_DELAY`. Điều này hợp lý về nghiệp vụ: một khách
> hàng từng trễ hạn nặng dù chỉ 1 lần vẫn đáng lo hơn một khách hàng trễ nhẹ đều đặn.
>
> `PAY_TO_BILL_RATIO` — tỷ lệ giữa `AVG_PAY_AMT` chia `AVG_BILL_AMT`, đo khách hàng trả
> được bao nhiêu phần trăm dư nợ trung bình. Ở đây em phải xử lý 2 vấn đề kỹ thuật: nếu
> `AVG_BILL_AMT` bằng 0 (khách không phát sinh dư nợ), phép chia sẽ lỗi — em thay 0
> bằng `NaN` trước khi chia, rồi điền lại 0 sau khi chia xong. Và vì tỷ lệ này có thể
> ra số cực lớn nếu mẫu số rất nhỏ, em giới hạn (`clip`) trong khoảng từ −10 đến 10, để
> tránh 1-2 điểm dữ liệu ngoại lai kéo lệch cả quá trình huấn luyện."

**Vì sao KHÔNG bỏ 6 cột BILL_AMT gốc sau khi đã có bản tổng hợp:**
> "Đây là câu hay bị hỏi — em GIỮ NGUYÊN cả 6 cột gốc lẫn 6 đặc trưng mới, không xóa
> bớt. Lý do: đa cộng tuyến chỉ thực sự gây hại cho mô hình tuyến tính như Logistic
> Regression; các mô hình cây như Random Forest, XGBoost, LightGBM hoàn toàn không bị
> ảnh hưởng bởi đa cộng tuyến, và đôi khi vẫn cần thông tin chi tiết theo từng tháng mà
> bản tổng hợp làm mất đi. Giữ cả 2 giúp mỗi mô hình tự chọn phần nó cần, thay vì em áp
> đặt trước phải bỏ bớt cột nào."

**Kết quả cuối:** 23 biến gốc → (đổi tên 1 cột, không đổi số lượng) → cộng thêm 6 biến
mới → 29 biến trước one-hot encoding → 35 chiều sau one-hot encoding 3 biến phân loại
(SEX, EDUCATION, MARRIAGE).

---

## Phụ lục D — Công thức thu hồi nợ (priority_score): quá trình thiết kế

> Đọc phần này nếu bị hỏi "công thức này lấy từ đâu ra, sao lại nhân, sao không cộng".

**Bước 1 — Xác định "ưu tiên" cần phản ánh điều gì:**
> "Em xuất phát từ câu hỏi: một khách hàng đáng được liên hệ thu hồi nợ TRƯỚC TIÊN khi
> nào? Em cho rằng cần đồng thời 2 điều kiện: (a) mô hình dự đoán khả năng vỡ nợ cao —
> đây là `risk_score`, và (b) đã có bằng chứng hành vi cụ thể — từng trễ hạn thật, chứ
> không chỉ là dự đoán suông. Nếu chỉ dùng riêng `risk_score`, danh sách ưu tiên sẽ gồm
> cả những khách hàng CHƯA từng trễ hạn ngày nào nhưng bị mô hình đánh giá rủi ro cao vì
> lý do khác — gọi điện cho nhóm này ngay có thể hơi sớm, khi họ chưa thực sự có dấu
> hiệu cụ thể."

**Bước 2 — Vì sao NHÂN 2 đại lượng, không CỘNG:**
> "Đây là quyết định thiết kế quan trọng nhất của công thức. Nếu em CỘNG
> `risk_score + delay_severity`, một khách hàng có `risk_score` rất cao nhưng
> `delay_severity` bằng 0 (chưa từng trễ hạn) vẫn nhận được điểm ưu tiên đáng kể — chỉ
> nhờ phần `risk_score` gánh. Còn khi em NHÂN, hễ MỘT trong 2 yếu tố bằng 0 thì
> `priority_score` LẬP TỨC bằng 0 — tạo ra hiệu ứng giống một phép 'VÀ' logic: chỉ
> khách hàng thỏa mãn CẢ 2 điều kiện mới được đẩy lên cao. Đây chính xác là hành vi em
> muốn, và em đã kiểm chứng bằng case study cụ thể ở slide 12 — nhóm rủi ro cao nhưng
> chưa trễ hạn có `priority_score` đúng bằng 0, bị loại khỏi danh sách liên hệ ngay."

**Bước 3 — Vì sao `delay_severity` lấy MAX chứ không phải tổng hay trung bình:**
> "Em dùng `max(PAY_1...PAY_6, 0)` — lấy giá trị lớn nhất trong 6 tháng, và chặn dưới ở
> 0. Lấy MAX vì em muốn phản ánh mức độ NGHIÊM TRỌNG NHẤT từng xảy ra — một khách hàng
> trễ 3 tháng đúng 1 lần đáng báo động hơn nhiều so với trễ 1 tháng đều đặn 3 lần, dù
> tổng hay trung bình có thể ra kết quả gần giống nhau. Chặn dưới ở 0 vì các giá trị âm
> trong `PAY_n` (như −1: trả đúng hạn, −2: không phát sinh dư nợ) không mang ý nghĩa
> 'mức độ rủi ro' — em không muốn một khách hàng có lịch sử tốt bị trừ điểm âm vô lý
> vào công thức nhân."

**Bước 4 — Kiểm chứng tính hợp lý (case study), không có nhãn để đánh giá bằng AUC:**
> "Điểm khác biệt quan trọng: công thức này KHÔNG được huấn luyện từ dữ liệu, vì không
> có nhãn nào ghi lại 'lẽ ra nên ưu tiên gọi ai trước' để em học theo — đây là luật do
> con người thiết kế (rule-based), không phải machine learning. Nên em không thể dùng
> AUC-ROC hay F1 để đánh giá nó. Cách em kiểm chứng là case study định tính: so sánh
> nhóm ưu tiên cao nhất (risk cao + đã trễ hạn 7-8 tháng) với nhóm risk cao nhưng chưa
> trễ hạn — xác nhận công thức phân biệt đúng 2 nhóm này như kỳ vọng thiết kế."

**Bước 5 — Hạn chế tự phát hiện khi lên tier:**
> "Khi chia thành 3 nhóm hành động theo phân vị 90%/60%, em phát hiện hơn 60% khách
> hàng có `delay_severity = 0`, khiến `priority_score` của họ cũng bằng 0 — và ngưỡng
> phân vị 60% vô tình trùng đúng vào khối giá trị 0 này, làm nhóm 'Thấp' bị rỗng. Đây
> là hạn chế em tìm ra khi phân tích số liệu thực tế, không phải giả định trước — và là
> hướng cần sửa: nên đặt ngưỡng theo giá trị `priority_score` tuyệt đối thay vì phân vị
> tương đối."

---

## Phụ lục E — Hyperparameters: từng tham số nghĩa là gì, vì sao chọn khoảng đó

> Đọc phần này nếu bị hỏi sâu ở slide 7 — "n_estimators là gì", "learning_rate ảnh
> hưởng thế nào", "vì sao chỉ thử 2-3 giá trị mỗi tham số".

**Logistic Regression — `C` ∈ {0,01; 0,1; 1,0; 10,0}:**
> "`C` là nghịch đảo cường độ chuẩn hóa (regularization). `C` càng NHỎ, mô hình bị ép
> giữ hệ số hồi quy càng gần 0 — chống overfitting nhưng dễ học chưa đủ (underfitting)
> nếu ép quá mạnh. `C` càng LỚN, mô hình được tự do khớp sát dữ liệu train hơn, nhưng dễ
> overfitting hơn. Em thử theo thang logarit — mỗi bước nhân 10 — vì ảnh hưởng của `C`
> lên mức độ chuẩn hóa là phi tuyến, thử theo cấp số nhân bao phủ được nhiều bậc độ lớn
> hơn là thử cộng dồn đều."

**Random Forest — `n_estimators` ∈ {200, 400}, `max_depth` ∈ {6, 10, None}:**
> "`n_estimators` là số lượng cây quyết định trong rừng. Nhiều cây hơn thường cho kết
> quả ổn định hơn (giảm phương sai, vì lấy trung bình phiếu bầu của nhiều cây), nhưng
> tốn thời gian huấn luyện hơn và lợi ích giảm dần sau một ngưỡng nhất định.
> `max_depth` là độ sâu tối đa mỗi cây được phép mọc — cây sâu hơn học được các tương
> tác phức tạp giữa nhiều biến hơn, nhưng dễ 'học vẹt' theo nhiễu của tập train nếu quá
> sâu. `None` nghĩa là không giới hạn — cây mọc tới khi mỗi lá chỉ còn 1 loại nhãn, đây
> là lựa chọn dễ overfit nhất trong 3 giá trị em thử."

**XGBoost / GBM / LightGBM — `n_estimators` ∈ {200, 400}, `max_depth` ∈ {3, 5},
`learning_rate` ∈ {0,05; 0,1}:**
> "3 mô hình này đều thuộc họ boosting — xây cây TUẦN TỰ, mỗi cây mới cố sửa lỗi mà các
> cây trước đó còn để sót, khác với Random Forest xây các cây độc lập rồi bầu chọn.
> `n_estimators` ở đây là số VÒNG boosting — số cây được thêm nối tiếp.
> `max_depth` trong boosting thường để NÔNG hơn Random Forest nhiều — 3 đến 5 tầng — vì
> mỗi cây chỉ cần học một phần nhỏ phần lỗi còn sót, không cần tự nó đã phức tạp.
> `learning_rate` là tốc độ học — mỗi cây mới đóng góp bao nhiêu phần trăm vào tổng dự
> đoán cuối cùng. `learning_rate` nhỏ (0,05) học chậm mà chắc, cần nhiều cây hơn nhưng
> ít overfit hơn; `learning_rate` lớn (0,1) học nhanh hơn nhưng dễ vọt qua điểm tối ưu.
> Đây là đánh đổi kinh điển trong boosting: giảm `learning_rate` thường nên tăng
> `n_estimators` để bù lại, và lưới của em có đủ cả 2 chiều để dò đánh đổi này."

**Neural Network (MLP) — `hidden_layer_sizes` ∈ {(32,), (64,32)}, `alpha` ∈ {1e-4, 1e-3}:**
> "`hidden_layer_sizes` là kiến trúc mạng — `(32,)` nghĩa là 1 tầng ẩn với 32 neuron,
> một mạng nông và đơn giản; `(64, 32)` là 2 tầng ẩn nối tiếp — 64 neuron rồi 32 neuron —
> mạng sâu hơn, biểu diễn được hàm phức tạp hơn nhưng cần nhiều dữ liệu hơn để không
> học vẹt. `alpha` là hệ số chuẩn hóa L2 cho trọng số mạng — giống `C` của Logistic
> Regression nhưng ngược chiều: `alpha` càng LỚN, chuẩn hóa càng MẠNH."

**Vì sao lưới chỉ có 2-3 giá trị mỗi tham số, không thử nhiều hơn:**
> "Đây là lựa chọn có chủ đích, không phải hạn chế kỹ thuật. Proposal ban đầu của em đã
> nêu 2 phương án: Optuna hoặc GridSearchCV — em chọn Grid Search với lưới nhỏ để đảm
> bảo tổng thời gian tinh chỉnh nằm trong ngân sách thời gian của đồ án (38 tổ hợp nhân
> 5 fold đã mất khoảng 24 phút). Ở phiên bản thử nghiệm sau (không thuộc bản chính thức
> hôm nay), em có mở rộng bằng Optuna với không gian tham số rộng hơn nhiều — kết quả
> cho thấy chỉ cải thiện thêm khoảng 0,16 điểm phần trăm AUC-ROC, xác nhận lưới nhỏ ban
> đầu đã gần đủ, không cần đầu tư thêm nhiều công sức vào việc mở rộng tìm kiếm."
