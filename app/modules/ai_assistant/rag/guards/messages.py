"""Thông điệp cố định (không qua LLM) cho các nhánh ngắt sớm."""

EMERGENCY_MESSAGE = (
    "Những gì bạn mô tả có thể là tình huống cấp cứu y tế. Hãy GỌI NGAY 115 hoặc đến khoa Cấp cứu gần nhất. "
    "Đừng tự điều trị hoặc chờ thêm. Nếu có người ở bên cạnh, hãy nhờ họ ở cùng và hỗ trợ gọi cấp cứu. "
    "Trợ lý AI không thể đánh giá tình trạng khẩn cấp thay cho nhân viên y tế."
)

SELF_HARM_MESSAGE = (
    "Mình rất tiếc khi nghe bạn đang trải qua điều này, và sự an toàn của bạn là quan trọng nhất lúc này. "
    "Nếu bạn đang có ý định làm hại bản thân hoặc đã uống/dùng chất gì đó nguy hiểm, hãy GỌI NGAY 115 hoặc đến "
    "cơ sở y tế gần nhất. Hãy nhắn hoặc gọi cho một người thân tin cậy để họ ở bên cạnh bạn ngay bây giờ."
)

OUT_OF_SCOPE_MESSAGE = (
    "Mình là Trợ lý Thông tin Y tế VitaP nên chỉ hỗ trợ các câu hỏi về sức khỏe, thuốc và chăm sóc y tế. "
    "Bạn có thể hỏi mình về thuốc, chỉ số sức khỏe hoặc một vấn đề sức khỏe cụ thể."
)

BLOCKED_MESSAGE = (
    "Mình không thể thực hiện yêu cầu này. Mình chỉ cung cấp thông tin y tế có dẫn nguồn và không thể bỏ qua "
    "các quy tắc an toàn của hệ thống."
)

MED_CHANGE_MESSAGE = (
    "Mình không thể tư vấn việc tự ý ngưng thuốc, tăng/giảm liều hoặc đổi cách dùng thuốc, vì quyết định này phụ "
    "thuộc vào tình trạng riêng của bạn. Hãy trao đổi với bác sĩ điều trị hoặc dược sĩ trước khi thay đổi. "
    "Nếu bạn đang gặp tác dụng phụ nặng hoặc dấu hiệu bất thường, hãy liên hệ cơ sở y tế ngay."
)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Mình chưa tìm thấy tài liệu đủ tin cậy trong kho tri thức để trả lời chính xác câu hỏi này, nên mình sẽ không "
    "đoán. Bạn có thể diễn đạt lại câu hỏi cụ thể hơn, hoặc hỏi bác sĩ/dược sĩ để được tư vấn."
)

AI_BUSY_MESSAGE = (
    "Hệ thống AI đang bận nên mình chưa thể trả lời lúc này. Vui lòng thử lại sau ít phút. "
    "Nếu câu hỏi liên quan đến sức khỏe của bạn và cần gấp, hãy liên hệ bác sĩ hoặc dược sĩ."
)

CLINICIAN_FALLBACK_MESSAGE = (
    "Mình không thể đưa ra chẩn đoán hay chỉ định thay đổi điều trị. Bạn nên trao đổi trực tiếp với bác sĩ điều trị "
    "hoặc dược sĩ để được tư vấn phù hợp với tình trạng của bạn."
)

PERSONAL_UNAVAILABLE_NOTE = (
    "Lưu ý: hiện chưa truy cập được hồ sơ sức khỏe của bạn nên câu trả lời này chưa được cá nhân hóa "
    "(dị ứng, thuốc đang dùng)."
)
