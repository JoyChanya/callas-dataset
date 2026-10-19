"""PII masking regex tests (acceptance criterion 4)."""
from src.clean import mask_pii


def test_mask_phone_dashes():
    assert mask_pii("โทร 081-234-5678 ด่วน") == "โทร [PHONE] ด่วน"


def test_mask_phone_plus66():
    assert "[PHONE]" in mask_pii("เบอร์ +66812345678")


def test_mask_id_card():
    assert mask_pii("เลข 1234567890123 ครับ") == "เลข [ID_CARD] ครับ"


def test_mask_account():
    out = mask_pii("โอนบัญชี 1234567890")
    assert out == "โอนบัญชี [ACCOUNT]"


def test_mask_url():
    assert mask_pii("กดลิงก์ http://bit.ly/x1Y นะ") == "กดลิงก์ [URL] นะ"


def test_org_name_kept():
    # organisation names carry scam signal — must NOT be masked
    txt = "นี่ธนาคารกสิกรไทยและสภ.เมือง"
    assert mask_pii(txt) == txt


def test_toggle_off():
    # disabling a masker leaves that entity untouched
    assert mask_pii("ดูที่ http://bit.ly/x1Y", {"mask_url": False}) == "ดูที่ http://bit.ly/x1Y"
