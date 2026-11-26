# Khi thống kê từng cặp chưa đủ để chọn điểm ghé sớm

Pilot này đo giá trị của hai lượt ghé trước khi điều xe dự phòng. Có $M$ cặp khách ứng viên, mọi khách đều phải được phục vụ trong ngày, và chi phí di chuyển bằng nhau giữa các lựa chọn ghé sớm. Xe được chọn khách thứ hai dựa trên nhu cầu của khách thứ nhất. Trên tuyến cần hỗ trợ, hai khách có pickup 0 hoặc 1 và xe chỉ còn một đơn vị sức chứa. Quá tải xảy ra khi tổng hai pickup lớn hơn 1.

## Mô hình

Mã hóa pickup thành dấu $-1,+1$. Viết $x=(z_1,\ldots,z_{2M},u,v)$, với $u,v$ là hai pickup trên tuyến cần hỗ trợ. Trong mô hình $j$, khách $2j-1$ và $2j$ tạo thành cặp hữu ích. Phân phối là

\[
P_j(x)=2^{-d}\left[1+\alpha\sum_{r<s}x_rx_s
+\eta z_{2j-1}z_{2j}uv\right],\qquad d=2M+2.
\]

Với $\alpha>0$, $0<\eta<1$ và $\alpha d/2+\eta<1$, mọi vector nhu cầu đều có xác suất dương. Thật vậy, tổng tích từng cặp bằng $((\sum_r x_r)^2-d)/2$, nên thấp nhất là $-d/2$. Khi lấy biên của tối đa ba pickup, tích bốn dấu triệt tiêu. Do đó mọi $P_j$ có cùng phân phối biên tới bậc ba và mọi cặp pickup có hệ số tương quan $\alpha$. Cặp biến gồm một pickup quan sát và nhãn quá tải $Y=\mathbf 1\{u=v=+1\}$ cũng có cùng phân phối giữa các mô hình.

## Giá trị tối ưu và giá trị minimax

Chi phí điều xe là $1+\alpha$. Nếu chờ, chi phí là $4Y$. Chi phí kỳ vọng khi chưa quan sát là $1+\alpha$ cho cả hai hành động. Với hai dấu quan sát $s,t$, gọi $r$ là hệ số tương tác của cặp được ghé. Lợi ích điều xe ở kết quả $(s,t)$, đã nhân xác suất của kết quả đó, bằng

\[
\frac{2\alpha(s+t)+(r-\alpha^2)st}{4}.
\]

Công thức này thu được từ $4P(Y=1,s,t)-(1+\alpha)P(s,t)$. Giá trị của một nhánh là phần dương của biểu thức, cộng trên các giá trị có thể thấy ở điểm thứ hai. Tối đa hóa riêng cho từng kết quả điểm đầu, rồi chọn điểm đầu tốt nhất, cho nghiệm tối ưu thích nghi.

Để tính minimax, xét phân phối trộn đều $M$ mô hình. Giá trị tối ưu dưới phân phối trộn là cận trên cho giá trị đảm bảo trong trường hợp xấu nhất. Ngẫu nhiên hóa một chính sách tối ưu của phân phối trộn bằng hoán vị đều các cặp đưa giá trị trong từng mô hình về đúng giá trị trung bình. Vì vậy cận trên đạt được. Script tính giá trị minimax trên họ $M$ mô hình bằng cách đặt $r=\eta/M$ ở mỗi cặp ứng viên.

Với $\alpha=\eta/(16M)$, cặp có tương tác là lựa chọn tốt nhất sau cả hai kết quả của lượt ghé đầu, vì $r>4\alpha+2\alpha^2$ cho cả $r=\eta$ lẫn $r=\eta/M$. Công thức suy ra

\[
G_{\text{biết mô hình}}=\frac{\eta-\alpha^2}{2},
\qquad
G_{\text{minimax}}=\frac{\eta/M-\alpha^2}{2}.
\]

Hai đại lượng đo tiền tiết kiệm kỳ vọng so với không quan sát. Ví dụ giữ tương quan ở mức $0.04$ với $M=8$ và $\eta=0.6$, giá trị biết mô hình là $0.2992$ và giá trị minimax là $0.05875$. Nếu đổi thứ tự tốn thêm $0.10$ đơn vị chi phí đã biết, người biết mô hình nên đổi. Người chỉ biết các biên chung có giá trị đảm bảo cao hơn khi giữ tuyến cũ.

## Cần bao nhiêu dữ liệu đồng bộ?

Một ngày quan sát cung cấp pickup của các khách ứng viên và một nhãn $Y$ cho cùng ngày. Chấm cặp $j$ bằng trung bình mẫu của $z_{2j-1}z_{2j}Y$. Kỳ vọng của điểm này bằng $\alpha/4$ ở cặp thường và $(\alpha+\eta)/4$ ở cặp hữu ích. Với các ngày độc lập, Hoeffding và union bound cho số ngày đủ để xác định đúng cặp với xác suất ít nhất $1-\delta$

\[
n\ge \frac{128}{\eta^2}\log\frac{2M}{\delta}.
\]

Về cận dưới, gọi $P_0$ là phân phối với $\eta=0$. Đặt $c=1-\alpha d/2>0$. Vì mật độ tương đối của $P_0$ thấp nhất là $c$, cận chi bình phương cho $D_{\rm KL}(P_j\|P_0)\le\eta^2/c$. Fano cho số ngày cần thiết để nhận đúng cặp với sai số tối đa $\delta$, kể cả khi được biết đầy đủ từng pickup theo ngày

\[
n\ge \frac{c}{\eta^2}\left[(1-\delta)\log M-\log 2\right].
\]

Với xác suất lỗi cố định và $c$ cách 0, hai cận khớp theo bậc $\log M/\eta^2$. Đây là số ngày quan sát **đồng bộ** trong mô hình. Pilot giả định có nhãn quá tải, chẳng hạn tính lại từ hai pickup thực tế. Nếu can thiệp làm mất cả hai pickup gốc thì cần cơ chế thu nhãn riêng.

## Chạy lại

```bash
python3 papers/informative_pickup_routing/pilot.py
```

Script kiểm bằng liệt kê đầy đủ 256 trạng thái cho ba cặp ứng viên, tính giá trị tối ưu thích nghi và chạy 256 lượt mô phỏng có seed cố định. Mô phỏng dùng phân phối tổng hợp có chủ đích. Mức tương quan co theo $M$ trong kết quả tiệm cận, và có thêm một điểm kiểm ở tương quan cố định $0.04$. Chưa có kết quả trên dữ liệu vận hành.

Hai tiền lệ cần đối chiếu sâu là tối ưu robust với các biên đã biết, và chọn đặc trưng thích nghi khi nhiều đặc trưng chỉ hữu ích nếu kết hợp. Phần cần kiểm tra novelty là giới hạn minimax cho giá trị ghé sớm khi quá tải phát sinh từ tổng pickup, cùng cận số ngày đồng bộ trong chính mô hình đó. [Distributionally Robust Linear and Discrete Optimization with Marginals](https://doi.org/10.1287/opre.2021.2243) · [When Does Synergy Help Active Feature Acquisition?](https://arxiv.org/abs/2609.32301)
