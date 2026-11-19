"""Fast (<50ms) rule-based Intent Router & Prompt Injection Guardrail for DUT AI."""

from dataclasses import dataclass
import re
import time
from typing import Optional


@dataclass(frozen=True)
class GuardrailDecision:
    """Outcome of evaluating an incoming user query against safety and domain rules."""

    allowed: bool
    intent: str
    category: Optional[str]
    reason: str
    refusal_message: Optional[str]
    latency_ms: float


# Specific DUT academic/administrative anchor phrases. Avoid generic words like
# "hướng dẫn", "quy định", or "sinh viên" alone so off-topic queries such as
# "Hướng dẫn giải tích phân" or "Java quản lý sinh viên" are not misclassified.
_DUT_DOMAIN_PATTERN = re.compile(
    r"("
    r"quy\s*chế\s*(đào\s*tạo|học\s*vụ|công\s*tác|tốt\s*nghiệp)?|"
    r"học\s*bổng|kkht|điểm\s*rèn\s*luyện|đrl|gpa|đtbchk|tín\s*chỉ|học\s*phần|"
    r"học\s*phí|miễn\s*giảm\s*học\s*phí|trợ\s*cấp\s*xã\s*hội|cảnh\s*báo\s*học|buộc\s*thôi\s*học|"
    r"xét\s*tốt\s*nghiệp|chuẩn\s*đầu\s*ra|toeic|ielts|đồ\s*án\s*tốt\s*nghiệp|khóa\s*luận|"
    r"bảo\s*lưu\s*kết\s*quả|nghỉ\s*học\s*tạm\s*thời|rút\s*bớt\s*học\s*phần|học\s*lại|học\s*cải\s*thiện|"
    r"phúc\s*khảo\s*điểm|khảo\s*thí|tuyển\s*sinh|nhập\s*học|"
    r"giấy\s*(xác|chứng)\s*nhận\s*sinh\s*viên|bảng\s*điểm|"
    r"ký\s*túc\s*xá|\bktx\b|bộ\s*phận\s*một\s*cửa|phòng\s*công\s*tác\s*sinh\s*viên|phòng\s*đào\s*tạo|"
    r"qđ-đhbk|tt-bgdđt|qđ-đhđn|văn\s*bản\s*pháp\s*quy|"
    r"kỷ\s*luật\s*sinh\s*viên|khiển\s*trách"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Priority-ordered guardrail rules. Jailbreak is evaluated unconditionally first.
_JAILBREAK_PATTERNS = (
    re.compile(
        r"(bỏ\s*qua\s*(tất\s*cả|mọi)\s*(các\s*)?(chỉ\s*dẫn|quy\s*tắc|lệnh|hướng\s*dẫn)|"
        r"hãy\s*quên\s*đi|quên\s*(hết|tất\s*cả)\s*(chỉ\s*dẫn|quy\s*tắc|vai\s*trò)|"
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?)|"
        r"forget\s+(all\s+)?(safety|previous|system)\s+(guidelines?|instructions?|rules?)|"
        r"disregard\s+(all\s+)?(previous|system)\s+(instructions?|prompts?)|"
        r"system\s*prompt|secret\s+instructions?|prompt\s*gốc|chỉ\s*thị\s*hệ\s*thống|"
        r"do\s*anything\s*now|\bDAN\b|developer\s*mode|jailbreak|"
        r"urgent\s*system\s*command|lệnh\s*khẩn\s*cấp|override\s*(system|safety|rules?)|"
        r"đóng\s*vai\s*(một\s*)?(hacker|tội\s*phạm|hiệu\s*trưởng)|"
        r"tình\s*huống\s*hư\s*cấu|in\s*ra\s*database|mật\s*khẩu\s*sinh\s*viên|"
        r"tấn\s*công\s*ddos|make\s+a\s+bomb|"
        r"lặp\s*lại\s*từ\s*['\"]?[^'\"]+['\"]?\s*liên\s*tục|"
        r"mọi\s*câu\s*hỏi.*đều\s*phải\s*trả\s*lời|"
        r"hãy\s*chửi\s*thề|nói\s*tục)",
        re.IGNORECASE | re.UNICODE,
    ),
)

_CODING_PATTERNS = (
    re.compile(
        r"(viết\s*(cho\s*mình\s*)?(code|chương\s*trình|script|hàm|câu\s*lệnh\s*sql|api)|"
        r"code\s*(giúp\s*mình|c\+\+|python|java|html|css)|"
        r"(?:^|[\s,:(])(c\+\+|python|java|javascript|typescript|html|css|sql|react|leetcode)(?:$|[\s,?.:)])|"
        r"bằng\s+c(?:$|[\s,.?!])|"
        r"lập\s*trình\s*hướng\s*đối\s*tượng|danh\s*sách\s*liên\s*kết|"
        r"thuật\s*toán\s*(dijkstra|sắp\s*xếp|tìm\s*kiếm|quy\s*hoạch\s*động)|"
        r"sửa\s*lỗi\s*(code|segmentation\s*fault)|segmentation\s*fault|debug\s*lỗi|compile\s*error)",
        re.IGNORECASE | re.UNICODE,
    ),
)

_STEM_HOMEWORK_PATTERNS = (
    re.compile(
        r"(tính\s*(tích\s*phân|đạo\s*hàm|giới\s*hạn|ma\s*trận|định\s*thức|công\s*lực|momen\s*quán\s*tính)|"
        r"giải\s*(giúp\s*mình\s*)?(chi\s*tiết\s*)?(phương\s*trình|bất\s*phương\s*trình|hệ\s*phương\s*trình|bài\s*tập|bài\s*toán|đề\s*thi|hộ\s*em\s*câu)|"
        r"tìm\s*cực\s*trị|chứng\s*minh\s*định\s*lý|"
        r"\b(sin|cos|tan|cot)\s*\([^)]*\)|"
        r"cân\s*bằng\s*phương\s*trình\s*hóa\s*học|định\s*luật\s*(newton|ohm|kirchhoff)|"
        r"mạch\s*điện\s*rlc|toán\s*rời\s*rạc)",
        re.IGNORECASE | re.UNICODE,
    ),
)

_EXTERNAL_LAW_PATTERNS = (
    re.compile(
        r"(vượt\s*đèn\s*đỏ|luật\s*giao\s*thông|nồng\s*độ\s*cồn|nghị\s*định\s*100|bằng\s*lái\s*xe|"
        r"làm\s*hộ\s*chiếu|cấp\s*lại\s*căn\s*cước\s*công\s*dân|cổng\s*dịch\s*vụ\s*công\s*quốc\s*gia|"
        r"bộ\s*luật\s*(hình\s*sự|dân\s*sự|lao\s*động)|"
        r"(thủ\s*tục|hồ\s*sơ\s*đăng\s*ký)\s*(ly\s*hôn|kết\s*hôn|sang\s*tên\s*xe)|"
        r"mức\s*lương\s*tối\s*thiểu\s*vùng|bảo\s*hiểm\s*xã\s*hội|lãi\s*suất\s*tiền\s*gửi|"
        r"thuế\s*thu\s*nhập\s*cá\s*nhân|luật\s*nghĩa\s*vụ\s*quân\s*sự)",
        re.IGNORECASE | re.UNICODE,
    ),
)

_CHITCHAT_PATTERNS = (
    re.compile(
        r"(quán\s*(bún|phở|cơm|ăn|cà\s*phê|trà\s*sữa)|ngon\s*rẻ|gần\s*trường.*quán|"
        r"thời\s*tiết\s*(hôm\s*nay|ngày\s*mai|đà\s*nẵng)|hôm\s*nay\s*ăn\s*gì|"
        r"câu\s*chuyện\s*cười|xả\s*stress|trận\s*bóng\s*đá|arsenal|chelsea|xổ\s*số|"
        r"nên\s*mua\s*iphone|samsung\s*s\d+|tỏ\s*tình|xem\s*bói|cung\s*hoàng\s*đạo|"
        r"phim\s*rạp|tư\s*vấn\s*tình\s*cảm|bạn\s*có\s*người\s*yêu\s*chưa|mấy\s*giờ\s*rồi)",
        re.IGNORECASE | re.UNICODE,
    ),
)

_REFUSAL_MESSAGES = {
    "jailbreak": (
        "Yêu cầu không hợp lệ do chứa dấu hiệu vượt quyền hoặc thay đổi chỉ thị hệ thống. "
        "Tôi chỉ hỗ trợ tra cứu quy chế, thủ tục hành chính và văn bản pháp quy của Trường ĐHBK – ĐHĐN."
    ),
    "coding": (
        "Xin lỗi, tôi là Trợ lý Học vụ chuyên tra cứu quy chế và thủ tục sinh viên DUT, "
        "không hỗ trợ viết code hoặc giải bài tập lập trình. Bạn có thể hỏi về quy chế đào tạo, học bổng hoặc thủ tục học vụ."
    ),
    "stem_homework": (
        "Xin lỗi, tôi không hỗ trợ giải bài tập môn học (Toán, Lý, Hóa...). "
        "Tôi chỉ giải đáp các quy chế đào tạo, học bổng, điểm rèn luyện và thủ tục hành chính tại Trường ĐHBK – ĐHĐN."
    ),
    "external_law": (
        "Xin lỗi, câu hỏi nằm ngoài phạm vi văn bản pháp quy nội bộ của Trường ĐHBK – ĐHĐN. "
        "Vui lòng tra cứu trên Cổng Dịch vụ công Quốc gia hoặc cơ quan chức năng có thẩm quyền."
    ),
    "chitchat": (
        "Xin chào! Tôi là Trợ lý Học vụ AI của Trường ĐHBK – ĐHĐN. "
        "Tôi chỉ hỗ trợ tra cứu quy chế, học bổng, lịch học vụ và thủ tục hành chính sinh viên."
    ),
}


def evaluate_query(query: str) -> GuardrailDecision:
    """Classify a query as in-domain DUT academic or blocked off-topic/injection."""
    started = time.perf_counter()
    cleaned = (query or "").strip()

    def _finish(allowed: bool, intent: str, category: Optional[str], reason: str) -> GuardrailDecision:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        refusal = None if allowed else _REFUSAL_MESSAGES.get(category or "", _REFUSAL_MESSAGES["chitchat"])
        return GuardrailDecision(
            allowed=allowed,
            intent=intent,
            category=category,
            reason=reason,
            refusal_message=refusal,
            latency_ms=elapsed_ms,
        )

    if not cleaned:
        return _finish(False, "invalid", "chitchat", "Empty query")

    # 1. Always block jailbreak / prompt injection regardless of any domain words.
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(cleaned):
            return _finish(False, "blocked", "jailbreak", "Matched prompt-injection / jailbreak pattern")

    has_dut_anchor = bool(_DUT_DOMAIN_PATTERN.search(cleaned))

    # 2. Check off-topic patterns; only skip coding/stem/external_law if query clearly
    #    asks about DUT academic regulations (e.g., "quy chế đồ án", "tín chỉ").
    category_rules = (
        ("coding", _CODING_PATTERNS, "Matched programming / coding homework pattern"),
        ("stem_homework", _STEM_HOMEWORK_PATTERNS, "Matched STEM homework pattern"),
        ("external_law", _EXTERNAL_LAW_PATTERNS, "Matched external civil/traffic law pattern"),
        ("chitchat", _CHITCHAT_PATTERNS, "Matched casual chitchat pattern"),
    )

    for category, patterns, reason in category_rules:
        if any(p.search(cleaned) for p in patterns):
            if has_dut_anchor and category != "chitchat":
                continue
            return _finish(False, "blocked", category, reason)

    if has_dut_anchor:
        return _finish(True, "dut_academic", None, "Matched DUT academic domain anchor")

    return _finish(True, "dut_general", None, "No off-topic or injection indicators detected")
