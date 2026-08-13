# Báo cáo tiến độ — Hệ thống Dự đoán Rủi ro Tín dụng & Đề xuất Thu hồi nợ

## Đối chiếu với Tiêu chí thành công đã cam kết trong proposal

---

## ✅ DONE

### 1. Dữ liệu & tiền xử lý
- 30.000 khách hàng, 25 cột, 0 giá trị thiếu/trùng lặp; tỷ lệ vỡ nợ 22,12% (6.636 vỡ nợ / 23.364 không vỡ nợ).
- Phát hiện đa cộng tuyến nghiêm trọng giữa `BILL_AMT1-6` (VIF 13,8–23,7, vượt ngưỡng cảnh báo 10) → giải quyết bằng 6 đặc trưng tổng hợp: `AVG_BILL_AMT`, `AVG_PAY_AMT`, `BILL_AMT_TREND`, `MAX_DELAY`, `AVG_DELAY`, `PAY_TO_BILL_RATIO`.
- Làm sạch `EDUCATION` (gộp 0/5/6 vào nhóm "khác"), `MARRIAGE` (gộp 0 vào nhóm "khác"), đổi tên `PAY_0`→`PAY_1`.
- Chia 60/20/20 (train/val/test, stratified) — 18.000/6.000/6.000 dòng, giữ đúng tỷ lệ 22,12% ở cả 3 tập. 29 đặc trưng trước one-hot, 35 sau one-hot.

### 2. Huấn luyện & tinh chỉnh — hành trình 7 phiên bản 
| Ver. | Thay đổi | Phát hiện chính |
|---|---|---|
| V1 | Baseline: 4 model, grid search, 1 validation split | AUC-ROC 0,7880 (XGBoost) |
| V2 | +GBM, +LightGBM (6 model), +PR-AUC/G-mean/Brier, thí nghiệm SMOTE riêng | SMOTE không đổi AUC-ROC nhưng làm lệch Brier score |
| V3 | Optuna (TPE), không gian tham số mở rộng, 40 trial/model | Chỉ +0,16pp AUC-ROC — xác nhận trần hiệu năng dataset |
| V4 | Grid + 5-fold Stratified CV (thay 1 validation set cố định) | Kết quả gần trùng V2, có thêm độ lệch chuẩn để đo ổn định |
| V5 | Chia 80/20 trực tiếp (thay 60/20/20 rồi gộp) | Tập test chỉ trùng 50,4% với V1-V4 → chênh AUC do đổi tập test (~0,9pp) **lớn hơn** chênh do đổi phương pháp tune (~0,16pp) |
| V6 | Bỏ hẳn thí nghiệm SMOTE, chỉ dùng class_weight | Chốt phương pháp xử lý mất cân bằng duy nhất |
| **V7** | Tính đủ 8 metric (không chỉ AUC-ROC) mỗi fold CV | Random Forest "mất" 3,67pp Recall nếu chỉ chọn tham số theo AUC-ROC |

### 3. Đánh giá trên tập Test 
- LightGBM tốt nhất: AUC-ROC 0,7886, nhưng chênh với XGBoost (0,7885)/GBM (0,7882) nhỏ hơn độ lệch chuẩn CV (~0,004) → 3 model này tương đương thống kê, không có "người thắng" rõ ràng.
- Confusion matrix LightGBM: TP=887, FN=440, FP=1.021, TN=3.652 → Recall 66,8%, Precision 46,5%.
- Lý giải bằng số học vì sao Precision "thấp": tỷ lệ nền 22% khiến Precision bị chặn trần toán học (base rate fallacy: cần Specificity ≥91,8% mới đạt Precision 70% ở cùng Recall) — không phải model kém.
- So với paper gốc Yeh & Lien (2009): LR cải thiện +3,55%, NN cải thiện +1,10%.
- So với paper Ampomah et al. (2025): phát hiện và chứng minh bằng số liệu (đối chiếu chính xác 10.572 non-default) lỗi rò rỉ dữ liệu do SMOTE chạy trước khi chia train/test — giải thích vì sao kết quả họ công bố (AUC 0,909) cao bất thường so với baseline không resample (0,737–0,751, gần khớp kết quả của dự án này).
- Đối chiếu Train(CV) vs Test: cả 6/6 model đều cho test tốt hơn CV nhẹ (do refit trên toàn bộ 24.000 dòng) — không có dấu hiệu overfitting.

### 4. Diễn giải & ứng dụng
- SHAP: `MAX_DELAY` > `LIMIT_BAL` > `PAY_1` là 3 yếu tố ảnh hưởng mạnh nhất; `MARRIAGE`/`SEX` cũng lọt top 15 (ghi nhận rủi ro fairness).
- `priority_score` giảm khối lượng liên hệ trực tiếp từ 6.000 → 600 khách hàng (top 10%), risk_score trung bình nhóm ưu tiên cao = 0,869.

### 5. Xử lý mất cân bằng dữ liệu 
- Thử nghiệm có kiểm soát: class_weight (baseline) vs SMOTE (thô) vs SMOTE+hiệu chỉnh xác suất (V2, Bước 5b) — kết luận bằng bằng chứng Brier score: SMOTE làm lệch xác suất dự đoán (do thay đổi tỷ lệ nền train từ 22% lên 50%), AUC-ROC không đổi nhưng Brier score xấu đi nếu không hiệu chỉnh lại.
- Quyết định: `class_weight`/`scale_pos_weight`/`sample_weight`/oversample thủ công (theo đúng API từng trong 6 model) là phương pháp xử lý mất cân bằng **chính thức duy nhất** (V6 trở đi) — không dùng SMOTE trong pipeline chính, tránh rủi ro lệch xác suất ảnh hưởng đến `priority_score`.


---

## ❌ NOT DONE — Công thức `priority_score`

1. Chưa tính giá trị dư nợ (exposure amount) — `risk_score × delay_severity` bỏ qua hoàn toàn quy mô tiền, 2 khách hàng cùng risk/delay nhưng dư nợ chênh lệch 100 lần vẫn xếp ưu tiên ngang nhau.
2. `delay_severity = max(PAY_1..6, 0)` chỉ lấy giá trị tệ nhất — chưa xét xu hướng (đang xấu đi/cải thiện) hay tần suất trễ hạn.
3. Ngưỡng phân nhóm tier (top 10%/30%/60%) đang bị lỗi thực tế — nhóm "Thấp" trống rỗng vì phần lớn khách hàng có `priority_score = 0` trùng đúng ngưỡng phân vị 60%.
4. Ngưỡng phân vị (10/30/60) chưa có cơ sở cost-benefit hay ràng buộc năng lực xử lý thực tế của bộ phận thu hồi nợ.
5. `priority_score` chưa từng được kiểm định bằng outcome thực tế (dataset không có dữ liệu theo dõi sau khi liên hệ).
6. Chưa kiểm tra calibration (Brier/reliability diagram) của chính pipeline chính (`class_weight`) — mới xác minh cho nhánh thí nghiệm SMOTE (mục DONE 5), chưa xác minh cho model thật sự dùng để tính `priority_score`.

---

## 🔜 TODO

### A. Hoàn thiện paper học thuật 

### B. Cải thiện công thức `priority_score`
4. Thêm biến giá trị dư nợ vào công thức: `priority_score = risk_score × delay_severity × exposure_amount` (hoặc dạng expected-loss chuẩn hơn).
5. Sửa lỗi ngưỡng tier "Thấp" trống — thiết kế lại ngưỡng phân nhóm (không dùng thuần phân vị khi phân phối có khối lượng lớn tại 0).
6. Thêm bước kiểm tra calibration (Brier score/reliability diagram) cho chính model dùng để tính `priority_score`, không chỉ nhánh thí nghiệm SMOTE.

### C. Tinh chỉnh xử lý mất cân bằng
7. Quét ngưỡng phân loại tối ưu theo F2-score/chi phí nghiệp vụ (không cần train lại, dùng thẳng `risk_score` đã có).
8. Đổi tiêu chí chọn `best_params` trong CV từ AUC-ROC sang F2/Recall, tận dụng `cv_all_results` đã có sẵn trong V7.
9. (Tuỳ chọn) thử SMOTE-Tomek/SMOTE-ENN kèm hiệu chỉnh xác suất, đối chiếu với kết luận đã có ở mục DONE 5.

### D. Mở rộng khác 
10. Xây dashboard Streamlit minh hoạ `priority_score`/tier tương tác (như mục 4.3 của paper CARE tham khảo) — không bắt buộc theo proposal (đã cho phép "notebook hoặc dashboard đơn giản"), nhưng sẽ tăng tính "ứng dụng thực tiễn" nếu làm.
11. Đánh giá fairness chính thức cho `MARRIAGE`/`SEX` (đã ghi nhận trong SHAP nhưng chưa phân tích sâu, vì đã lược khỏi bản paper rút gọn).
12. Pin version cụ thể cho `requirements*.txt`; chạy đa seed để kiểm tra độ ổn định tổng thể của toàn pipeline (không chỉ std nội bộ CV).

---

