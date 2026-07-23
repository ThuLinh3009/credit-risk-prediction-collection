# Hệ thống Dự đoán Rủi ro Tín dụng và Đề xuất Thu hồi nợ

Đối chiếu và mở rộng nghiên cứu gốc của **Yeh & Lien (2009)** trên bộ dữ liệu UCI *Default of Credit Card Clients*.

> Xây dựng mô hình Machine Learning dự báo xác suất vỡ nợ có độ chính xác cao và khả năng diễn giải, đồng thời chuyển hóa kết quả dự báo thành công cụ hỗ trợ ra quyết định thực tiễn cho hoạt động thu hồi nợ.

- **Người thực hiện:** Đoàn Thị Thu Linh
- **Thời gian:** 6 tuần (3 sprint × 2 tuần)
- **Ngày đề xuất:** 23/07/2026

---

## 1. Tổng quan

Bộ dữ liệu *Default of Credit Card Clients* (UCI Machine Learning Repository) gồm **30.000 khách hàng** thẻ tín dụng tại Đài Loan, với **23 biến đầu vào** (nhân khẩu học, hạn mức tín dụng, lịch sử thanh toán 6 tháng gần nhất, dư nợ và số tiền thanh toán) và biến mục tiêu nhị phân `default.payment.next.month` (tỷ lệ vỡ nợ ~22%, dữ liệu mất cân bằng).

Bộ dữ liệu được công bố cùng nghiên cứu gốc của Yeh & Lien (2009), so sánh 6 kỹ thuật phân loại (KNN, Logistic Regression, Discriminant Analysis, Naive Bayes, Neural Network, Classification Tree), trong đó Neural Network và Logistic Regression cho kết quả tốt nhất tại thời điểm đó.

Bên cạnh mục tiêu học thuật (đối chiếu paper gốc), dự án hướng đến giá trị ứng dụng thực tiễn: xây dựng công thức xếp hạng ưu tiên **`priority_score`** kết hợp xác suất vỡ nợ (`risk_score`) và mức độ trễ hạn thanh toán thực tế, giúp bộ phận thu hồi nợ tập trung nguồn lực vào nhóm khách hàng vừa rủi ro cao vừa đã trễ hạn nhiều kỳ, thay vì liên hệ dàn trải toàn bộ danh sách khách hàng.

## 2. Mục tiêu

**Mục tiêu tổng quát:** xây dựng và đánh giá một hệ thống dự báo rủi ro vỡ nợ có độ chính xác cao, có khả năng diễn giải, đồng thời chuyển hóa kết quả dự báo thành công cụ hỗ trợ ra quyết định cho hoạt động thu hồi nợ.

**Mục tiêu cụ thể:**

1. Xây dựng mô hình ML dự báo xác suất vỡ nợ, so sánh tối thiểu 3–4 thuật toán: Logistic Regression, Random Forest, XGBoost và Neural Network.
2. Đối chiếu kết quả với Yeh & Lien (2009): xác định các mô hình hiện đại có vượt trội hơn Neural Network 2009 hay không, và mức cải thiện là bao nhiêu %.
3. Xác định top đặc trưng ảnh hưởng mạnh nhất đến khả năng vỡ nợ bằng SHAP / feature importance.
4. Xây dựng công thức xếp hạng ưu tiên thu hồi nợ `priority_score = risk_score × mức độ trễ hạn`, cùng bảng danh sách khách hàng xếp hạng theo mức độ ưu tiên cần liên hệ.
5. Hoàn thiện báo cáo/paper, slide thuyết trình và một demo minh họa (notebook hoặc dashboard đơn giản).

## 3. Phạm vi dự án

**Dữ liệu:** bộ dữ liệu công khai UCI *Default of Credit Card Clients* — 30.000 quan sát, 23 biến đầu vào, 1 biến mục tiêu nhị phân, ~22% khách hàng vỡ nợ.

**Thuật toán áp dụng:**

| Mô hình | Vai trò |
|---|---|
| Logistic Regression | Baseline, đối chiếu trực tiếp với paper gốc |
| Random Forest | Đại diện nhóm ensemble bagging |
| XGBoost / LightGBM | Đại diện nhóm ensemble boosting, hiệu năng cao trên dữ liệu dạng bảng |
| Neural Network (MLP) | Đối chiếu trực tiếp với mô hình Neural Network trong paper 2009 |

**Ngoài phạm vi:** không triển khai hệ thống production/API thời gian thực; không tích hợp CRM/hệ thống thu hồi nợ thực tế (demo dừng ở notebook/dashboard minh họa); không thu thập thêm dữ liệu ngoài bộ UCI đã công bố.

## 4. Phương pháp luận

Quy trình triển khai theo chuẩn CRISP-DM:

```
Tìm hiểu bài toán & paper gốc
        │
        ▼
Tìm hiểu & làm sạch dữ liệu
        │
        ▼
Feature engineering
        │
        ▼
Huấn luyện & tinh chỉnh mô hình
        │
        ▼
Đánh giá & diễn giải (SHAP)
        │
        ▼
Xây dựng công cụ xếp hạng ưu tiên
        │
        ▼
Báo cáo & trình bày
```

### Công thức xếp hạng ưu tiên thu hồi nợ

```
priority_score = risk_score × mức độ trễ hạn
```

- `risk_score`: xác suất vỡ nợ (0–1) do mô hình tốt nhất dự báo.
- `mức độ trễ hạn`: suy ra từ các biến `PAY_1`–`PAY_6` (ví dụ: giá trị trễ hạn lớn nhất trong 6 kỳ gần nhất, hoặc trung bình có trọng số theo thời gian gần).

Công thức được thiết kế sơ bộ ở Sprint 1, hoàn thiện và kiểm định qua case study cụ thể ở Sprint 2 và Sprint 3.

### Công cụ & môi trường

- **Ngôn ngữ & thư viện:** Python, pandas, scikit-learn, XGBoost/LightGBM, SHAP, Optuna (hoặc GridSearchCV)
- **Môi trường thực nghiệm:** Jupyter Notebook

## 5. Kế hoạch triển khai

### Sprint 1 (Tuần 1–2) — Data Understanding & Preparation

- Đọc và tóm tắt paper gốc Yeh & Lien (2009): phương pháp, kết quả, cách so sánh 6 kỹ thuật.
- EDA: thống kê mô tả, phân phối biến mục tiêu, kiểm tra mức độ mất cân bằng (~22% default).
- Làm sạch dữ liệu: xử lý giá trị bất thường ở `EDUCATION` (0, 5, 6), `MARRIAGE` (0), `PAY_n` (-2); đổi tên `PAY_0` thành `PAY_1` để nhất quán.
- Feature engineering: mã hóa biến phân loại, chuẩn hóa biến số, xử lý đa cộng tuyến giữa `BILL_AMT1`–`BILL_AMT6`.
- Chia tập train/validation/test theo tỷ lệ phù hợp, stratify theo biến mục tiêu.
- Phác thảo sơ bộ công thức `priority_score` dựa trên `PAY_0`–`PAY_6`.

### Sprint 2 (Tuần 3–4) — Model Building, Training & Rule-based Recommendation

- Baseline Logistic Regression, đối chiếu trực tiếp với paper gốc.
- Huấn luyện Random Forest, XGBoost/LightGBM, Neural Network (MLP); xử lý mất cân bằng bằng class-weight.
- Hyperparameter tuning bằng GridSearch/Optuna cho từng mô hình.
- Đánh giá bằng AUC-ROC, Precision, Recall, F1, Confusion Matrix; chọn mô hình tốt nhất, xuất risk score cho từng khách hàng.
- Hoàn thiện `priority_score = risk_score × max(PAY_0...PAY_6)`, chạy thử trên tập validation, kiểm tra tính hợp lý của thứ hạng.

### Sprint 3 (Tuần 5–6) — Evaluation, Interpretation & Reporting

- So sánh AUC/Accuracy giữa mô hình hiện đại và kết luận của Yeh & Lien (2009).
- Diễn giải mô hình: SHAP/feature importance, xác định yếu tố ảnh hưởng mạnh nhất đến vỡ nợ.
- Tinh chỉnh ngưỡng `priority_score`, kiểm tra qua case study cụ thể.
- Xây dựng bảng xếp hạng ưu tiên thu hồi nợ (dashboard đơn giản hoặc bảng trong notebook).
- Viết báo cáo cuối: phương pháp, kết quả, so sánh paper gốc, đề xuất chính sách thu hồi nợ, hạn chế, hướng phát triển.
- Chuẩn bị slide thuyết trình và demo minh họa.

## 6. Cấu trúc thư mục

```
VSF-Fintech-Track3/
├── datasets/           # Dữ liệu UCI Default of Credit Card Clients (.csv / .xls)
├── docs/               # Đề xuất dự án (proposal), tài liệu bổ sung
├── notebooks/          # Notebook EDA, feature engineering, huấn luyện & đánh giá mô hình
├── papers/             # Paper gốc Yeh & Lien (2009) và tài liệu tham khảo
├── presentations/      # Slide thuyết trình
├── src/                # Mã nguồn tái sử dụng (tiền xử lý, huấn luyện, tính priority_score)
└── README.md
```

## 7. Sản phẩm bàn giao

- Notebook/mã nguồn xử lý dữ liệu, huấn luyện và đánh giá mô hình (tái lập được kết quả).
- Bảng so sánh hiệu năng 3–4 thuật toán, đối chiếu với kết quả paper gốc Yeh & Lien (2009).
- Phân tích SHAP/feature importance chỉ ra các yếu tố ảnh hưởng mạnh nhất đến vỡ nợ.
- Công thức `priority_score` cùng bảng danh sách khách hàng xếp hạng theo mức độ ưu tiên thu hồi nợ.
- Báo cáo/paper hoàn chỉnh và slide thuyết trình kèm demo minh họa (notebook hoặc dashboard).

## 8. Tiêu chí thành công

| Tiêu chí | Ngưỡng kỳ vọng |
|---|---|
| Số thuật toán so sánh | Tối thiểu 3–4 thuật toán, đầy đủ AUC-ROC, Precision, Recall, F1 |
| Cải thiện so với paper gốc | Xác định rõ mô hình tốt nhất có/không vượt trội hơn Neural Network 2009, kèm % cải thiện cụ thể |
| Khả năng diễn giải | Xác định top đặc trưng ảnh hưởng mạnh nhất, có SHAP plot minh họa |
| Công cụ ưu tiên thu hồi nợ | `priority_score` hoạt động ổn định, thứ hạng đầu ra hợp lý qua case study |
| Tài liệu & trình bày | Báo cáo, slide và demo hoàn chỉnh, trình bày độc lập không cần giải thích thêm |

## 9. Rủi ro và biện pháp giảm thiểu

| Rủi ro | Biện pháp giảm thiểu |
|---|---|
| Dữ liệu mất cân bằng (~22% default) làm mô hình thiên lệch | Dùng class-weight, đánh giá bằng AUC-ROC/F1 thay vì chỉ Accuracy, cân nhắc resampling nếu cần |
| Đa cộng tuyến giữa `BILL_AMT1`–`BILL_AMT6` | Kiểm tra VIF/hệ số tương quan, cân nhắc giảm chiều hoặc loại bỏ biến dư thừa trước khi huấn luyện Logistic Regression |
| So sánh không công bằng với paper gốc do khác cách đánh giá (Lift vs AUC) | Quy đổi/đối chiếu cẩn thận phương pháp đánh giá gốc, nêu rõ giả định khi so sánh |
| Thời gian tinh chỉnh siêu tham số vượt kế hoạch | Ưu tiên Optuna với ngân sách thử nghiệm giới hạn (time-boxed) |
| Công thức `priority_score` chưa phản ánh đúng nghiệp vụ thực tế | Kiểm tra qua nhiều case study cụ thể, sẵn sàng điều chỉnh trọng số/ngưỡng dựa trên phản hồi định tính |

## 10. Cài đặt

```bash
git clone <repository-url>
cd VSF-Fintech-Track3
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install pandas scikit-learn xgboost lightgbm shap optuna jupyter
```

## 11. Tham khảo

- Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.
- UCI Machine Learning Repository — [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## Tác giả

**Đoàn Thị Thu Linh**
