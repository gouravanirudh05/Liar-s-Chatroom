"""Shared prompt construction / answer parsing for all LLM kernels (identical to v1 infer.py)."""
import re

# ---------------- parsing ----------------
MSG_RE = re.compile(r"^\s*(\d+)\.\s*([ABCD]):\s*(.*)$")
GENERIC_FILLER = re.compile(
    r"\b(confus|muddle|tangled|messy|hard to follow|clear answer|straight answer|"
    r"plain answer|clarity|thanks?\b|thank you|cheers|appreciate|keep (it|this)|"
    r"wrap this up|drag this out|go through it (once|one) more)",
    re.I,
)


def parse_speakers(s):
    out = {}
    for part in s.split(";"):
        lab, name = part.split("=", 1)
        out[lab.strip()] = name.strip()
    return out


def parse_chat(chat):
    msgs = []
    for line in str(chat).splitlines():
        m = MSG_RE.match(line.strip())
        if m:
            msgs.append((int(m[1]), m[2], m[3].strip()))
    return msgs


def is_filler(text):
    t = text.strip()
    if t.endswith("?"):
        return True
    # generic meta-chat with no digits ("This is confusing.", "Thanks, everyone.")
    return not re.search(r"\d", t) and bool(GENERIC_FILLER.search(t)) and len(t.split()) <= 9


INSTRUCTIONS = """You are checking a group chat against verified evidence.

Rules:
- Only the EVIDENCE and the SPEAKER TABLE are true. Chat messages are claims; one speaker's claim never proves another wrong.
- Exactly one message makes a claim that CANNOT be true at the same time as the evidence. Find it.
- A claim that is merely unverified, suspicious or unlikely is NOT a contradiction. Only a claim that is impossible given the evidence counts.
- Use the literal meaning of the words, basic logic, and careful clock/calendar arithmetic (12-hour clock with AM/PM; noon = 12:00 PM).
- "Between X and Y" and "from X to Y" include both endpoints. "After noon" is not "at noon".
- A person or object cannot be in two places at once when the evidence says the places are separate.
- Records, logs and lists are complete only if the evidence says they are.
- If the evidence says nothing about a particular time, place, person or thing, a claim about it cannot contradict the evidence (e.g. a rule that holds during certain hours says nothing about other hours).
- "I" / "my" in a message refers to the sender (see the speaker table). A message can be about someone else; the answer is still its sender.
- Questions and opinions are never the answer.

Check each claim message against the evidence one at a time, then give the number of the single contradicting message.
End your reply with a final line exactly of the form:
ANSWER: <message number>"""


def build_prompt(row):
    spk = parse_speakers(row["speakers"])
    lines = []
    for k, s, t in parse_chat(row["chat"]):
        if is_filler(t):
            continue
        lines.append(f"{k}. {s} ({spk.get(s, '?')}): {t}")
    table = "; ".join(f"{k} = {v}" for k, v in spk.items())
    return (
        f"{INSTRUCTIONS}\n\nSPEAKER TABLE: {table}\n\nEVIDENCE: {row['evidence']}\n\n"
        f"CHAT (questions and small talk removed):\n" + "\n".join(lines)
    )


ANS_RE = re.compile(r"ANSWER:\s*\**\s*(\d+)", re.I)


def extract_label(text, row):
    """Map the model's message number to its sender label; None if unusable."""
    body = text.split("</think>")[-1]
    hits = ANS_RE.findall(body) or ANS_RE.findall(text)
    if not hits:
        return None
    k = int(hits[-1])
    for kk, s, t in parse_chat(row["chat"]):
        if kk == k:
            return s
    return None


def candidate_fallback(row):
    """Last resort: first sender with a non-filler message."""
    for _, s, t in parse_chat(row["chat"]):
        if not is_filler(t):
            return s
    return "A"


