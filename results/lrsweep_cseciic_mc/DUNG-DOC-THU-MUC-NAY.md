# ⛔ Thư mục này chạy SAI CẤU HÌNH. Đừng dùng số ở đây.

240 run này dùng `downsample_per_class = 130000`, thừa kế im lặng từ cấu hình nền
`e1_botiot.yaml`. Bản đã nộp cho ô CSE-CIC **đa lớp** dùng **50000**.

Kết quả là hai thí nghiệm khác nhau, và chúng cho kết luận ngược nhau:

| downsample | KAN-8 trừ MLP-PM-80 (TB 5 vòng cuối) |
|---|---|
| 130000 (thư mục này, SAI) | **−1,24 pp** |
| 50000 (đúng, xem `../lrsweep_cseciic_mc50k/`) | **+1,38 pp** |

Giữ lại thay vì xoá vì hai lý do. Một, log thô là dữ liệu: xoá đi thì lần sau không ai
kiểm được rằng chuyện này đã xảy ra. Hai, nó là bằng chứng cho một giả thuyết chưa kiểm
là **lợi thế đa lớp của KAN thu hẹp khi có thêm dữ liệu**. Muốn dùng cho mục đích đó thì
phải thiết kế thí nghiệm riêng, không được lấy hai thư mục này ghép thành một xu hướng.

Số dùng cho bài nằm ở `../lrsweep_cseciic_mc50k/`.
