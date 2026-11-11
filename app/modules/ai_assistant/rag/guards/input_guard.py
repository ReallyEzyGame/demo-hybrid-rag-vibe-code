"""LỚP 1 — Input Guardrail (thiết kế §2.2). Thuần regex đã chuẩn hóa Unicode, không gọi LLM, mục tiêu < 20ms."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ...enums import AnswerStatus, WarningLevel
from .messages import (
    BLOCKED_MESSAGE,
    EMERGENCY_MESSAGE,
    MED_CHANGE_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    SELF_HARM_MESSAGE,
)
from .text_utils import fold, lower_nfc


class GuardAction(StrEnum):
    PASS = "PASS"
    SHORT_CIRCUIT = "SHORT_CIRCUIT"


class Hint(StrEnum):
    NONE = "NONE"
    DIAGNOSIS_REQUEST = "DIAGNOSIS_REQUEST"


@dataclass(frozen=True)
class GuardDecision:
    action: GuardAction
    reason_code: str = "OK"
    answer_status: AnswerStatus | None = None
    warning_level: WarningLevel = WarningLevel.INFO
    message: str = ""
    hint: Hint = Hint.NONE

    @property
    def short_circuit(self) -> bool:
        return self.action is GuardAction.SHORT_CIRCUIT


def _compile(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


# --- Mẫu áp lên văn bản đã bỏ dấu (khớp cả câu có dấu lẫn không dấu) ---
_EMERGENCY_FOLDED = _compile([
    r"dau nguc (du doi|khung khiep|nhu bi (de|bop|ep))",
    r"tuc nguc.{0,30}kho tho", r"kho tho.{0,30}tuc nguc",
    r"kho tho (cap|du doi|nang|nhieu|dot ngot)",
    r"khong tho (duoc|noi)", r"tho khong ra hoi", r"ngung tho", r"tim tai",
    r"ho ra mau", r"non ra mau", r"chay mau (nhieu|o at|khong cam)",
    r"meo (mieng|mom)", r"liet (nua nguoi|mot ben|tay|chan)", r"te liet",
    r"dot ngot.{0,30}(te |liet|yeu |meo|mat y thuc|ngat|hoa mat|noi kho)",
    r"co giat", r"bat tinh", r"hon me", r"ngat xiu",
    r"soc phan ve", r"phan ve", r"phu (ne )?(co hong|luoi|moi)", r"sung (co hong|luoi|moi).{0,30}kho tho",
    r"uong qua lieu", r"qua lieu thuoc", r"ngo doc (thuoc|cap)",
    r"uong (nham|qua|lo) ?\d{2,} ?(vien|goi|ong)",
    r"(nuot|uong) (ca )?(lo|vi|hop) thuoc",
    r"uong nham.{0,60}(non|buon non|chong mat|kho tho|lo mo)",
    r"(non|buon non).{0,40}uong nham",
])
# Mẫu tự hại không dấu, rõ nghĩa (không có 'tu tu' vì trùng 'từ từ')
_SELF_HARM_FOLDED = _compile([
    r"tu sat", r"muon chet", r"khong muon song nua", r"ket lieu (doi|ban than|chinh)",
    r"uong bao nhieu.{0,25}(thi )?(chet|tu vong)", r"lieu (gay )?(chet|tu vong)",
])
# --- Mẫu nhạy dấu (áp lên văn bản giữ dấu) ---
_ACCENT_SENSITIVE = _compile([
    r"tự tử", r"tự sát", r"tự hại", r"tự làm đau", r"muốn chết", r"không muốn sống", r"chấm dứt cuộc (sống|đời)",
    r"nói đớ", r"nói ngọng đột ngột", r"méo miệng",
])
_SELF_HARM_ACCENT = _compile([r"tự tử", r"tự sát", r"tự hại", r"tự làm đau", r"muốn chết", r"không muốn sống", r"chấm dứt cuộc (sống|đời)"])

_INJECTION = _compile([
    r"ignore (all |any )?(the )?(previous|prior|above|earlier|your) (instructions?|rules?|prompts?)",
    r"disregard .{0,20}(instructions?|rules?)", r"forget (all |your )?(previous |prior )?(instructions?|rules?)",
    r"bo qua (moi |cac |tat ca |het )?(quy tac|huong dan|chi thi|quy dinh|rang buoc|gioi han)",
    r"quen (het |tat ca )?(quy tac|huong dan)",
    r"(system|developer) prompt", r"prompt he thong", r"huong dan he thong",
    r"(tiet lo|cho toi xem|in ra|noi cho toi).{0,25}(prompt|chi thi)",
    r"\bjailbreak\b", r"\bdan mode\b", r"developer mode", r"gia vo (la|nhu).{0,30}(khong co|bo) (gioi han|quy tac)",
    r"drop\s+table", r"delete\s+from", r"truncate\s+table", r"insert\s+into", r"union\s+select",
    r"select .{0,40} from .{0,40}(users?|password|chat_)", r";\s*--", r"rm\s+-rf",
    r"xoa (bang|co so du lieu|database).{0,30}(users?|nguoi dung|he thong)",
    r"(viet|tao).{0,12}(code|ma nguon|script|lenh).{0,40}(xoa|drop|hack|tan cong|bypass)",
])
_HARMFUL = _compile([
    r"che tao (chat doc|bom|vu khi|thuoc no|ma tuy)",
    r"(dieu che|tong hop|san xuat) (ma tuy|heroin|methamphetamine|cocaine|chat doc|thuoc no|vu khi)",
    r"vu khi sinh hoc", r"cach (giet|dau doc|ha doc) (nguoi|ai do)",
])
_MED_CHANGE = _compile([
    r"(tu y|tu minh|tu) (bo|ngung|dung|giam|tang|doi|cat) (uong )?(thuoc|lieu)",
    r"toi (muon|dinh|se|co nen) (bo|ngung|dung|giam|tang|doi|cat) (uong )?(thuoc|lieu)",
    r"(bo|ngung|dung|cat) (uong )?thuoc .{0,40}(duoc khong|duoc chua|co sao khong|co nen khong|co sao ko)",
    r"(tang|giam|doi|gap doi) lieu .{0,40}(duoc khong|duoc chua|co nen|co sao)",
    r"co nen (bo|ngung|dung|giam|tang|doi) (uong )?(thuoc|lieu)",
])
_DIAGNOSIS = _compile([
    r"(la|bi|mac) (benh|chung) gi", r"benh gi (vay|a|day|the|nhi)", r"(chan doan|doan benh) (giup|cho toi|xem)",
    r"toi (bi|mac) (benh )?gi\b", r"co phai toi (bi|mac)",
])
_NON_MEDICAL = _compile([
    r"\b(python|javascript|typescript|java|html|css|lap trinh|viet code|thuat toan|giai (phuong trinh|toan|bai tap)|"
    r"chinh tri|bau cu|tong thong|bong da|world cup|ca sy|choi game|chung khoan|bitcoin|crypto|gia vang|ty gia|"
    r"thoi tiet|du lich|nau an|cong thuc nau|dich (sang|cau)|lam tho|viet (bai|email|thu|van|tho))\b",
])
_MEDICAL_MARKERS = (
    "thuoc", "benh", "suc khoe", "trieu chung", "xet nghiem", "huyet ap", "duong huyet", "nhip tim", "can nang",
    "bmi", "spo2", "nhiet do", "sot", "dieu tri", "bac si", "duoc si", "benh vien", "vitamin", "khang sinh",
    "viem", "tieu duong", "tim mach", "ung thu", "di ung", "tac dung phu", "lieu dung", "chan doan", "phau thuat",
    "mang thai", "thai ky", "dinh duong", "hba1c", "a1c", "cholesterol", "dau ", "met moi", "ho ", "vet thuong",
)

_MAX_INPUT_CHARS = 5000  # chặn đầu vào quá dài trước khi chạy regex


def _short(code: str, status: AnswerStatus, level: WarningLevel, message: str) -> GuardDecision:
    return GuardDecision(GuardAction.SHORT_CIRCUIT, code, status, level, message)


class InputGuard:
    def evaluate(self, text: str) -> GuardDecision:
        text = text[:_MAX_INPUT_CHARS]
        folded = fold(text)
        accented = lower_nfc(text)

        # 1A. Cấp cứu / tự hại — luôn ưu tiên cao nhất
        if _SELF_HARM_FOLDED.search(folded) or _SELF_HARM_ACCENT.search(accented):
            return _short("SELF_HARM", AnswerStatus.EMERGENCY_REDIRECT, WarningLevel.CRITICAL, SELF_HARM_MESSAGE)
        if _EMERGENCY_FOLDED.search(folded) or _ACCENT_SENSITIVE.search(accented):
            return _short("MEDICAL_EMERGENCY", AnswerStatus.EMERGENCY_REDIRECT, WarningLevel.CRITICAL, EMERGENCY_MESSAGE)

        # 1B. Prompt injection / nội dung độc hại
        if _INJECTION.search(folded):
            return _short("INJECTION", AnswerStatus.OUT_OF_SCOPE, WarningLevel.INFO, BLOCKED_MESSAGE)
        if _HARMFUL.search(folded):
            return _short("HARMFUL_CONTENT", AnswerStatus.OUT_OF_SCOPE, WarningLevel.INFO, BLOCKED_MESSAGE)

        # Yêu cầu tự ý đổi liều / ngưng thuốc (§1.4-3): từ chối cố định, không gọi LLM
        if _MED_CHANGE.search(folded):
            return _short("MEDICATION_CHANGE_REQUEST", AnswerStatus.NEEDS_CLINICIAN, WarningLevel.WARNING, MED_CHANGE_MESSAGE)

        # Ngoài phạm vi: chỉ khi có chủ đề phi y tế rõ ràng VÀ không có dấu hiệu y tế nào
        if _NON_MEDICAL.search(folded) and not any(m in folded for m in _MEDICAL_MARKERS):
            return _short("NON_MEDICAL", AnswerStatus.OUT_OF_SCOPE, WarningLevel.INFO, OUT_OF_SCOPE_MESSAGE)

        hint = Hint.DIAGNOSIS_REQUEST if _DIAGNOSIS.search(folded) else Hint.NONE
        return GuardDecision(GuardAction.PASS, hint=hint)
