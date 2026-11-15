"""Exact rule modules for known evidence families.

Each module looks for its own evidence pattern; if found, it inspects the chat
and returns the message numbers it can PROVE contradict the evidence. Modules
must be high-precision: when unsure, flag nothing. `solve` returns a label only
when the flagged messages all come from exactly one sender.
"""
import re

T = r"(\d{1,2}:\d{2} [AP]M)"
NEG = re.compile(r"\b(not|never|no one|nobody|didn't|did not|wasn't|weren't|hadn't|haven't|isn't)\b|n't\b", re.I)


def tmin(s):
    h, m, ap = re.match(r"(\d{1,2}):(\d{2}) ([AP]M)", s).groups()
    h, m = int(h) % 12, int(m)
    return (h + (12 if ap == "PM" else 0)) * 60 + m


def times(text):
    return [tmin(x) for x in re.findall(T, text)]


def dur(s):
    """'1 hour 30 minutes' / '45 minutes' / '2 hours' -> minutes."""
    h = re.search(r"(\d+) hours?", s)
    m = re.search(r"(\d+) minutes?", s)
    return (int(h[1]) * 60 if h else 0) + (int(m[1]) if m else 0)


DUR = r"((?:\d+ hours?)?(?: ?\d+ minutes?)?)"


def parse_speakers(s):
    out = {}
    for part in s.split(";"):
        lab, name = part.split("=", 1)
        out[lab.strip()] = name.strip()
    return out


MSG_RE = re.compile(r"^\s*(\d+)\.\s*([ABCD]):\s*(.*)$")


def parse_chat(chat):
    out = []
    for line in str(chat).splitlines():
        m = MSG_RE.match(line.strip())
        if m:
            out.append((int(m[1]), m[2], m[3].strip()))
    return out


def single_time(text):
    ts = times(text)
    return ts[0] if len(ts) == 1 else None


# ---------------------------------------------------------------- modules
def m_no_entry_window(ev, msgs, spk):
    """'Nobody could enter the X between T1 and T2.' -> entering X at t in [T1,T2]."""
    flags = []
    for place, a, b in re.findall(r"Nobody could enter the ([\w\s-]+?) between " + T + " and " + T, ev):
        lo, hi = tmin(a), tmin(b)
        for k, s, t in msgs:
            tl = t.lower()
            if place.lower() not in tl or NEG.search(t) or t.endswith("?"):
                continue
            if not re.search(r"\b(enter|entered|into)\b", tl):
                continue
            x = single_time(t)
            if x is not None and lo <= x <= hi:
                flags.append(k)
    return flags


OFF_PATTERNS = [
    r"(?:the )?([\w\s-]+?) was (?:off|stopped|shut off|offline|switched off) from " + T + " to " + T,
    r"From " + T + " to " + T + r",? the ([\w\s-]+?) was (?:off|stopped|shut off|offline|switched off)",
]


def m_only_while(ev, msgs, spk):
    """'The X could <do> only while the Y was on' + 'Y off from T1 to T2'."""
    flags = []
    for subj, verb, src in re.findall(
        r"The ([\w\s-]+?) could ([\w\s-]+?) only while the ([\w\s-]+?) was (?:on|running|working)", ev
    ):
        windows = []
        for pat in OFF_PATTERNS:
            for g in re.findall(pat, ev):
                if pat.startswith("From"):
                    a, b, name = g
                else:
                    name, a, b = g
                if name.lower().strip().split()[-1] in src.lower():
                    windows.append((tmin(a), tmin(b)))
        if not windows:
            continue
        key = subj.lower().split()[-1]  # e.g. 'lamps', 'lift', 'machine'
        srcw = src.lower().split()[-1]  # e.g. 'generator', 'power', 'supply'
        for k, s, t in msgs:
            tl = t.lower()
            if key not in tl or srcw in tl or NEG.search(t) or t.endswith("?"):
                continue
            x = single_time(t)
            if x is not None and any(lo <= x <= hi for lo, hi in windows):
                flags.append(k)
    return flags


def m_vpn(ev, msgs, spk):
    """Remote sign-in requires VPN; VPN offline from T1 to T2."""
    if "required the VPN" not in ev:
        return []
    w = re.findall(r"VPN service was offline from " + T + " to " + T, ev)
    if not w:
        return []
    lo, hi = tmin(w[0][0]), tmin(w[0][1])
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if "remote" in tl and "sign" in tl and not NEG.search(t):
            x = single_time(t)
            if x is not None and lo <= x <= hi:
                flags.append(k)
    return flags


BOARD = re.compile(r"\b(get on|got on|getting on|board|boarded|boarding|climb\w* onto|step\w* onto|go aboard|went aboard|climb\w* aboard)\b", re.I)


def m_vehicle_away(ev, msgs, spk):
    """Vehicle left stop at T1 and could not be back before T2 -> boarding there in (T1, T2)."""
    flags = []
    specs = []
    for veh, stop, a, b in re.findall(
        r"The ([\w\s-]+?) left the ([\w\s-]+?) at " + T + r" and did not return before " + T, ev
    ):
        specs.append((veh, stop, tmin(a), tmin(b)))
    for veh, stop, a, d in re.findall(
        r"The ([\w\s-]+?) left the ([\w\s-]+?) at " + T + r"\. Its round trip took at least " + DUR + ",", ev
    ):
        specs.append((veh, stop, tmin(a), tmin(a) + dur(d)))
    for veh, stop, lo, hi in specs:
        for k, s, t in msgs:
            tl = t.lower()
            if veh.lower() not in tl or stop.lower() not in tl or not BOARD.search(t) or NEG.search(t):
                continue
            x = single_time(t)
            if x is not None and lo < x < hi:
                flags.append(k)
    return flags


def m_percent_sales(ev, msgs, spk):
    """Exactly N <items> were sold ..., exactly p% before noon."""
    flags = []
    for n, item, p in re.findall(
        r"Exactly (\d+) (\w+) were sold for [\w\s'-]+?, and exactly (\d+)% of them were sold before noon", ev
    ):
        n, p = int(n), int(p)
        before = n * p / 100
        after = n - before
        for k, s, t in msgs:
            m = re.search(r"(\d+) " + item + r"\b", t)
            if not m or NEG.search(t):
                continue
            v = int(m[1])
            tl = t.lower()
            if "before noon" in tl and v != before:
                flags.append(k)
            elif "after noon" in tl and v != after:
                flags.append(k)
    return flags


def m_codes(ev, msgs, spk):
    """Before T only code c1 could unlock the D; from T to end of day only code c2."""
    m1 = re.search(r"Before " + T + r",? only code (\d+) could unlock the ([\w\s-]+?)\.", ev)
    m2 = re.search(r"From " + T + r" to the end of the day, only code (\d+) could unlock it", ev)
    if not (m1 and m2) or m1[1] != m2[1]:
        return []
    t0, c1, c2 = tmin(m1[1]), m1[2], m2[2]
    flags = []
    for k, s, t in msgs:
        if "unlock" not in t.lower() or NEG.search(t):
            continue
        cm = re.findall(r"code (\d+)", t)
        x = single_time(t)
        if len(cm) != 1 or x is None:
            continue
        ok = c1 if x < t0 else c2
        if cm[0] != ok:
            flags.append(k)
    return flags


MODULES = [m_no_entry_window, m_only_while, m_vpn, m_vehicle_away, m_percent_sales, m_codes]


def solve(row):
    """Return (label or None, {module: [flagged msg numbers]})."""
    ev = row["evidence"]
    msgs = parse_chat(row["chat"])
    spk = parse_speakers(row["speakers"])
    sender = {k: s for k, s, _ in msgs}
    hits = {}
    for mod in MODULES:
        try:
            f = mod(ev, msgs, spk)
        except Exception:  # a parse failure on an unfamiliar case means "no opinion"
            f = []
        if f:
            hits[mod.__name__] = f
    senders = {sender[k] for f in hits.values() for k in f}
    return (senders.pop() if len(senders) == 1 else None), hits


# ================================================================ batch 2
import datetime as _dt

MONTHS = {m: i + 1 for i, m in enumerate(
    "January February March April May June July August September October November December".split())}
MONTH_RE = "(" + "|".join(MONTHS) + ")"
DATE = r"(\d{1,2} " + MONTH_RE[1:-1].join(["(?:", ")"]) + r" \d{4})"
WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
NUMWORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def pdate(s):
    d, m, y = s.split()
    return _dt.date(int(y), MONTHS[m], int(d))


def dates(text):
    return [pdate(x) for x in re.findall(DATE, text)]


def num(s):
    s = s.lower()
    return NUMWORDS[s] if s in NUMWORDS else int(s)


STOPCAPS = set("""At The Only From Nobody Before It Its Every Exactly All There Until On In During Everyone
One Room Codes Code That Each Nothing Starting Throughout Leaving For Between Ever After Then Who Honestly Okay
So Come By Yes No This Some Thanks Cheers Another Both Neither Not Later Earlier Just Well Also A An I My
Everything Let Is Was Were Do Does Could Should Would Can Has Have Did Are Please Someone Anyone Things""".split())


def subject(t, sender_name, names):
    """Who a claim is about: a named person if exactly one is mentioned, else the sender if 'I'.

    Returns None when unsure (e.g. 'I saw the supervisor ...' with no name)."""
    mentioned = [n for n in names if n not in STOPCAPS and re.search(r"\b" + re.escape(n) + r"\b", t)]
    if len(mentioned) == 1:
        return mentioned[0]
    if mentioned:
        return None
    if re.search(r"\b(saw|see|seen|watched|spotted|noticed)\b", t, re.I):
        return None
    if re.search(r"\b(I|my|me)\b", t):
        return sender_name
    return None


def all_names(ev, spk):
    caps = set(re.findall(r"\b([A-Z][a-z]+)\b", ev))
    return set(spk.values()) | {c for c in caps if c not in MONTHS and c not in WEEKDAYS and c not in STOPCAPS}


def m_ages(ev, msgs, spk):
    born = {p: pdate(d) for p, d in re.findall(r"([A-Z][a-z]+) was born on " + DATE, ev)}
    if not born:
        return []
    flags = []
    for k, s, t in msgs:
        if NEG.search(t) or t.endswith("?"):
            continue
        who = subject(t, spk.get(s), set(born) | set(spk.values()))
        if who not in born:
            continue
        b = born[who]
        rest = re.sub(DATE, " ", t)
        rest = re.sub(MONTH_RE + r" \d{4}", " ", rest)
        ages = [int(x) for x in re.findall(r"\b(\d{1,3})\b", rest)]
        if len(ages) != 1:
            continue
        age = ages[0]
        ds = dates(t)
        if "turned" in t and not ds:
            mm = re.search(MONTH_RE + r" (\d{4})", t)
            if not mm:
                continue
            ok = MONTHS[mm[1]] == b.month and int(mm[2]) - b.year == age
        elif len(ds) == 1 and "turned" not in t:
            d = ds[0]
            ok = d.year - b.year - ((d.month, d.day) < (b.month, b.day)) == age
        else:
            continue
        if not ok:
            flags.append(k)
    return flags


def m_rota(ev, msgs, spk):
    m = re.search(r"From " + DATE + " to " + DATE + r", exactly one person worked the ([\w\s-]+?) each night", ev)
    if not m:
        return []
    d0, d1, duty = pdate(m[1]), pdate(m[2]), m[3]
    assign = {}
    # form (a): 'Eun on 13–15 September' / 'Obi on 16 September'
    for p, a, b, mon in re.findall(r"([A-Z][a-z]+) on (\d{1,2})(?:[–-](\d{1,2}))? " + MONTH_RE, ev):
        a, b = int(a), int(b or a)
        for day in range(a, b + 1):
            assign[_dt.date(d0.year, MONTHS[mon], day)] = p
    # form (b): 'Lena took the first two nights, Reza took the next two, and Theo took the last three'
    if not assign:
        parts = re.findall(r"([A-Z][a-z]+) took the (?:first|next|last) (\w+)", ev)
        cur = d0
        for p, n in parts:
            for _ in range(num(n)):
                assign[cur] = p
                cur += _dt.timedelta(days=1)
        if parts and cur - _dt.timedelta(days=1) != d1:
            return []  # parse mismatch: be safe
    flags = []
    for k, s, t in msgs:
        if duty.lower() not in t.lower() or NEG.search(t) or not re.search(r"\bI\b", t):
            continue
        ds = dates(t)
        if len(ds) == 1 and ds[0] in assign and assign[ds[0]] != spk.get(s):
            flags.append(k)
    return flags


def _weekday_fn(ev):
    m = re.search(DATE + r" was a (" + "|".join(WEEKDAYS) + ")", ev)
    if not m:
        return None
    anchor, wd = pdate(m[1]), WEEKDAYS.index(m[2])
    return lambda d: (wd + (d - anchor).days) % 7


def m_opening(ev, msgs, spk):
    wdf = _weekday_fn(ev)
    m = re.search(r"the ([\w\s-]+?) was open only (\w+day) to (\w+day)(?: in " + MONTH_RE + r" \d{4})?, from "
                  + T + " to " + T, ev, re.I)
    if not (wdf and m):
        return []
    place = m[1].lower()
    a, b = WEEKDAYS.index(m[2]), WEEKDAYS.index(m[3])
    open_days = {(a + i) % 7 for i in range((b - a) % 7 + 1)}
    lo, hi = tmin(m[5]), tmin(m[6])
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if place not in tl or NEG.search(t) or "outside" in tl or "waiting" in tl or t.endswith("?"):
            continue
        ds, x = dates(t), single_time(t)
        if len(ds) != 1 or x is None:
            continue
        if wdf(ds[0]) not in open_days or not lo <= x <= hi:
            flags.append(k)
    return flags


def m_scan_entry(ev, msgs, spk):
    """Entry only by scanning own active <card>; P's card inactive before T."""
    m = re.search(r"Nobody could enter the ([\w\s-]+?) except by scanning their own active ([\w\s-]+?)\.", ev)
    if not m:
        return []
    place = m[1].lower()
    inactive = {p: tmin(x) for p, x in re.findall(r"([A-Z][a-z]+)'s [\w\s-]+? was inactive before " + T, ev)}
    names = all_names(ev, spk)
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if place not in tl or NEG.search(t) or "outside" in tl or not re.search(r"\b(enter\w*|into)\b", tl):
            continue
        who = subject(t, spk.get(s), names)
        x = single_time(t)
        if who in inactive and x is not None and x < inactive[who]:
            flags.append(k)
    return flags


def m_route(ev, msgs, spk):
    m = re.search(r"The ([\w\s-]+?) and the ([\w\s-]+?) are separate locations, and every route from the "
                  r"[\w\s-]+? to the [\w\s-]+? takes at least (\d+) minutes", ev)
    if not m:
        return []
    l1, l2, d = m[1], m[2], int(m[3])
    anchors = re.findall(r"([A-Z][a-z]+) was at the " + re.escape(l1) + " at " + T, ev)
    if not anchors:
        return []
    names = all_names(ev, spk)
    flags = []
    for p, a in anchors:
        t0 = tmin(a)
        for k, s, t in msgs:
            tl = t.lower()
            if "at the " + l2.lower() not in tl or NEG.search(t) or "road" in tl or "way" in tl:
                continue
            who = subject(t, spk.get(s), names)
            x = single_time(t)
            if who == p and x is not None and abs(x - t0) < d:
                flags.append(k)
    return flags


def m_ownership(ev, msgs, spk):
    m = re.search(r"Nobody else owned the ([\w\s-]+?) in " + MONTH_RE + r" (\d{4})", ev)
    if not m:
        return []
    item, mon, yr = m[1], MONTHS[m[2]], int(m[3])
    sales = []
    for a, b, d in re.findall(r"([A-Z][a-z]+) owned the [\w\s-]+? until (?:selling|s?he sold|they sold) it to ([A-Z][a-z]+) on " + DATE, ev):
        sales.append((a, b, pdate(d)))
    for a, b, d in re.findall(r"([A-Z][a-z]+) (?:then )?sold it to ([A-Z][a-z]+) on " + DATE, ev):
        sales.append((a, b, pdate(d)))
    for b, d, a in re.findall(r"Until selling it to ([A-Z][a-z]+) on " + DATE + r",? ([A-Z][a-z]+) owned", ev):
        sales.append((a, b, pdate(d)))
    for d, a, b in re.findall(r"On " + DATE + r",? ([A-Z][a-z]+) sold it to ([A-Z][a-z]+)", ev):
        sales.append((a, b, pdate(d)))
    if not sales or re.search(r"days after buying", ev):
        return []
    sales = sorted(set(sales), key=lambda x: x[2])
    if len(sales) != len(re.findall(r"\b(sold|selling)\b", ev)):
        return []  # some sale not parsed: be safe
    for (a1, b1, _), (a2, _, _) in zip(sales, sales[1:]):
        if b1 != a2:
            return []  # chain not contiguous: be safe

    def owner(d):
        if any(d == x[2] for x in sales):
            return None  # sale day: ambiguous
        o = sales[0][0]
        for a, b, sd in sales:
            if d > sd:
                o = b
        return o

    flags = []
    for k, s, t in msgs:
        if item.lower() not in t.lower() or NEG.search(t) or not re.search(r"\bI\b", t):
            continue
        ds = dates(t)
        if len(ds) != 1 or (ds[0].year, ds[0].month) != (yr, mon):
            continue
        d, me = ds[0], spk.get(s)
        if re.search(r"\bI (?:\w+ )?owned\b", t) or re.search(r"\bI owned\b", t):
            o = owner(d)
            if o and o != me:
                flags.append(k)
        elif re.search(r"\bI sold\b", t):
            to = re.search(r"to ([A-Z][a-z]+)", t) or re.search(r"I sold ([A-Z][a-z]+) the", t)
            if to and (me, to[1], d) not in sales:
                flags.append(k)
        elif re.search(r"\bI bought\b", t):
            fr = re.search(r"from ([A-Z][a-z]+)", t)
            if fr and (fr[1], me, d) not in sales:
                flags.append(k)
    return flags


MODULES += [m_ages, m_rota, m_opening, m_scan_entry, m_route, m_ownership]


# ================================================================ batch 3
NAME = r"([A-Z][a-z]+)"
ARRIVE_CLAIM = re.compile(r"\bfirst\b", re.I)


def m_arrivals(ev, msgs, spk):
    """First-arrival times: direct facts, door-log lists and 'exactly N minutes after/before' chains."""
    known, rel = {}, []
    log = re.search(r"first arrived at the [\w\s-]+? that day: (.+?)\.(?: |$)", ev)
    if log:
        for p, x in re.findall(NAME + " at " + T, log[1]):
            known[p] = tmin(x)
    for p, x in re.findall(NAME + r" first arrived at the [\w\s-]+? at " + T, ev):
        known[p] = tmin(x)
    for clause in re.split(r"(?<=\.) |, and |; ", ev):
        m = re.search(r"Exactly (\d+) minutes (after|before) " + NAME + r", " + NAME + " first arrived", clause)
        if m:
            rel.append((m[4], m[3], int(m[1]) * (1 if m[2] == "after" else -1)))
            continue
        m = re.search(NAME + r" first arrived exactly (\d+) minutes (after|before) " + NAME, clause)
        if m:
            rel.append((m[1], m[4], int(m[2]) * (1 if m[3] == "after" else -1)))
    n_rel_mentions = len(re.findall(r"minutes (after|before)", ev))
    if len(rel) != n_rel_mentions or not (known or rel):
        return []
    for _ in range(6):
        for subj, ref, off in rel:
            if ref in known and subj not in known:
                known[subj] = known[ref] + off
            if subj in known and ref not in known:
                known[ref] = known[subj] - off
    flags = []
    for k, s, t in msgs:
        me = spk.get(s)
        if not ARRIVE_CLAIM.search(t) or NEG.search(t) or t.endswith("?") or me not in known:
            continue
        others = [n for n in known if n != me and re.search(r"\b" + n + r"\b", t)]
        x = single_time(t)
        if x is not None and not others and re.search(r"\b(I|my)\b", t, re.I):
            if re.search(r"\b(before|after)\b", t):
                continue
            if x != known[me]:
                flags.append(k)
        elif x is None and len(others) == 1 and others[0] in known:
            o = others[0]
            if re.search(r"\bafter\b", t) and not known[me] > known[o]:
                flags.append(k)
            elif re.search(r"\bbefore\b", t) and not known[me] < known[o]:
                flags.append(k)
    return flags


def _level_model(ev):
    """Return (container, unit, f(t)->level or None) for single-container level families."""
    bases = re.findall(r"(?:The|the) ([\w\s-]+?) held (?:exactly )?(\d+) (\w+) at " + T, ev)
    bases += [(c, n, u, x) for x, c, n, u in re.findall(r"At " + T + r",? the ([\w\s-]+?) held (?:exactly )?(\d+) (\w+)", ev)]
    if len(bases) != 1:
        return None
    cont, n0, unit, x0 = bases[0]
    n0, t0 = float(n0), tmin(x0)
    events = []  # (time, delta)
    for x, kk, u, op in re.findall(r"[Aa]t " + T + r",? exactly (\d+) (\w+) were (added|removed)", ev):
        events.append((tmin(x), float(kk) * (1 if op == "added" else -1)))
    for kk, u, op, x in re.findall(r"[Ee]xactly (\d+) (\w+) were (added|removed) at " + T, ev):
        events.append((tmin(x), float(kk) * (1 if op == "added" else -1)))
    events = sorted(set(events))
    if len(events) != len(re.findall(r"\b(added|removed) at|were (added|removed)", ev)) - 0 and \
            len(events) != len(re.findall(r"were (added|removed)", ev)):
        return None
    rates = []  # (start, end, per-minute rate)
    m = re.search(r"and (drained|filled) at a steady (\d+) " + unit + r" per hour until " + T, ev)
    if m:
        rates.append((t0, tmin(m[3]), float(m[2]) / 60 * (-1 if m[1] == "drained" else 1)))
    for a, b, op, r in re.findall(r"[Ff]rom " + T + " to " + T + r",? it (filled|drained) at a steady (\d+) " + unit + " per hour", ev):
        rates.append((tmin(a), tmin(b), float(r) / 60 * (-1 if op == "drained" else 1)))
    n_rates = len(re.findall(r"at a steady", ev))
    if len(rates) != n_rates:
        return None
    if rates:
        end = max(b for _, b, _ in rates)
        if events:
            return None
    else:
        e = re.search(r"Nothing else was added or removed before " + T, ev)
        if e:
            end = tmin(e[1])
        elif re.search(r"Nothing else was added or removed\.", ev):
            end = 24 * 60
        else:
            return None
    ev_times = {x for x, _ in events}

    def level(t):
        if t < t0 or t > end or t in ev_times:
            return None
        v = n0 + sum(d for x, d in events if x < t)
        for a, b, r in rates:
            if t > a:
                v += r * (min(t, b) - a)
        return v

    return cont, unit, level


def m_levels(ev, msgs, spk):
    mod = _level_model(ev)
    if not mod:
        return []
    cont, unit, level = mod
    key = cont.lower().split()[-1]
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if key not in tl or unit not in tl or NEG.search(t) or t.endswith("?"):
            continue
        if re.search(r"\b(added|removed|took|put|poured|filled|drained|emptied)\b", tl):
            continue
        x = single_time(t)
        rest = re.sub(T, " ", t)
        vals = re.findall(r"(\d+(?:\.\d+)?) " + unit, rest)
        if x is None or len(vals) != 1:
            continue
        lv = level(x)
        if lv is not None and abs(lv - float(vals[0])) > 1e-6:
            flags.append(k)
    return flags


def m_holder(ev, msgs, spk):
    """One item passed between holders; nobody else held it within the window."""
    w = re.search(r"Nobody else held the ([\w\s-]+?) between " + T + " and " + T, ev)
    if not w or "only one person held" not in ev.lower():
        return []
    item, ws, we = w[1], tmin(w[2]), tmin(w[3])
    seq = []  # (holder, end)
    for m in re.finditer(NAME + r" held it (?:from " + T + " to " + T + r"|until " + T + ")"
                         r"|(?:handed it to|then to) " + NAME + r"(?:, who held it until " + T + ")?", ev):
        if m[1]:
            end = m[3] or m[4]
            if seq and seq[-1][0] == m[1] and seq[-1][1] is None:
                seq[-1] = (m[1], tmin(end))
            else:
                seq.append((m[1], tmin(end)))
        else:
            seq.append((m[5], tmin(m[6]) if m[6] else None))
    if not seq or any(e is None for _, e in seq) or seq[-1][1] != we:
        return []
    bounds, start = [], ws
    for p, e in seq:
        bounds.append((p, start, e))
        start = e

    def holder(t):
        for p, a, b in bounds:
            if a < t < b:
                return p
        return None

    names = all_names(ev, spk)
    key = item.lower().split()[-1]
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if key not in tl or NEG.search(t) or t.endswith("?"):
            continue
        if not re.search(r"\b(held|holding|had|in my hands|hands)\b", tl) or re.search(r"\b(put|gave|handed|left)\b", tl):
            continue
        who = subject(t, spk.get(s), names)
        x = single_time(t)
        if who and x is not None:
            h = holder(x)
            if h and h != who:
                flags.append(k)
    return flags


def m_bookings(ev, msgs, spk):
    m = re.search(r"([Tt]he [\w\s-]+?|Room \d+) was booked (?:without gaps )?from " + T + r"(?: to " + T + r")?(?: without gaps)?:? (.+?)\. During", ev)
    m2 = re.search(r"(?:From|Starting from|Starting at) " + T + r",? (the [\w\s-]+?|Room \d+) was booked without gaps: (.+?)\. During", ev)
    if m:
        room, st, en, body = m[1], tmin(m[2]), (tmin(m[3]) if m[3] else None), m[4]
    elif m2:
        room, st, en, body = m2[2], tmin(m2[1]), None, m2[3]
    else:
        return []
    room = re.sub(r"^[Tt]he ", "", room)
    segs = re.findall(r"the (\w+) team(?: (?:until " + T + r"|for " + DUR + r"))?", body)
    out, cur = [], st
    for i, (team, until, d) in enumerate(segs):
        if until:
            end = tmin(until)
        elif d.strip():
            end = cur + dur(d)
        elif i == len(segs) - 1 and en is not None:
            end = en
        else:
            return []
        out.append((team, cur, end))
        cur = end
    if en is not None and cur != en:
        return []
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        tm = re.search(r"the (\w+) team", tl)
        if not tm or room.lower() not in tl or NEG.search(t) or t.endswith("?"):
            continue
        x = single_time(t)
        if x is None:
            continue
        for team, a, b in out:
            if a < x < b and team.lower() != tm[1]:
                flags.append(k)
    return flags


def m_travel(ev, msgs, spk):
    flags = []
    for m in re.finditer(r"The ([\w\s-]+?) left the [\w\s-]+? at " + T + r" and went straight (?:along the (\d+) km route )?to the ([\w\s-]+?)[.:,]", ev):
        veh, t0, dist, dest = m[1], tmin(m[2]), m[3], m[4]
        tail = ev[m.end():m.end() + 250]
        if dist:
            sp = re.search(r"(?:speed never went above|never went faster than|never went above) (\d+) km/h", tail)
            if not sp:
                continue
            mins = float(dist) / float(sp[1]) * 60
        else:
            legs = re.findall(r"(\d+) km where it never went above (\d+) km/h", tail)
            if not legs:
                continue
            mins = sum(float(d) / float(v) * 60 for d, v in legs)
        earliest = t0 + mins
        for k, s, t in msgs:
            tl = t.lower()
            if veh.lower() not in tl or dest.lower() not in tl or NEG.search(t) or "another" in tl or "way" in tl:
                continue
            if not re.search(r"\b(got to|arrived|reached|pulled into|made it to|get to)\b", tl):
                continue
            x = single_time(t)
            if x is not None and x < earliest - 1e-9:
                flags.append(k)
    return flags


def m_only_present(ev, msgs, spk):
    specs = []
    for a, b, p, q, place in re.findall(r"From " + T + " to " + T + r",? only " + NAME + " and " + NAME + r" were at the ([\w\s-]+?)[.,]", ev):
        specs.append((tmin(a), tmin(b), {p, q}, place))
    for p, q, place, a, b in re.findall(r"Only " + NAME + " and " + NAME + r" were at the ([\w\s-]+?) from " + T + " to " + T, ev):
        specs.append((tmin(a), tmin(b), {p, q}, place))
    if not specs or not re.search(r"[Nn]obody else", ev):
        return []
    names = all_names(ev, spk)
    flags = []
    for lo, hi, ok, place in specs:
        for k, s, t in msgs:
            tl = t.lower()
            if "at the " + place.lower() not in tl or NEG.search(t) or re.search(r"\b(away|left|outside|past|near|leaving)\b", tl):
                continue
            who = subject(t, spk.get(s), names)
            x = single_time(t)
            if who and x is not None and lo <= x <= hi and who not in ok:
                flags.append(k)
    return flags


def m_register(ev, msgs, spk):
    m = re.search(r"(?:register|list)[\w\s:-]*? (?:lists exactly \w+ people|\w+ people appear on [\w\s-]+?)(?: for the " + T + r" [\w\s-]+?| at " + T + r")?: ([^.]+)\.", ev)
    if not m or "everyone who attended is on it" not in ev:
        return []
    listed = set(re.findall(NAME, m[3]))
    flags = []
    for k, s, t in msgs:
        me = spk.get(s)
        tl = t.lower()
        if t.endswith("?") or not re.search(r"\bI\b", t) or len(re.findall(NAME, t.replace("I ", ""))) > 0:
            continue
        neg = bool(re.search(r"\b(wasn't|was not|skipped|missed|didn't attend|did not attend|not at)\b", tl))
        pos = bool(re.search(r"\b(attended|was at|went to|sat in on|joined)\b", tl)) and not neg
        if (neg and me in listed) or (pos and me not in listed):
            flags.append(k)
    return flags


MODULES += [m_arrivals, m_levels, m_holder, m_bookings, m_travel, m_only_present, m_register]


# ================================================================ batch 4
ENTER = re.compile(r"\b(enter\w*|into|came in|come in|went in|got in)\b", re.I)


def m_gate_group(ev, msgs, spk):
    """Before T the door opened only for group G and was the only way in; known non-members can't enter before T."""
    pats = [
        r"Before " + T + r",? the ([\w\s-]+?) opened only for ([\w\s-]+?), and it was the only way into the ([\w\s-]+?)\.",
        r"The ([\w\s-]+?) was the only way into the ([\w\s-]+?), and before " + T + r",? it opened only for ([\w\s-]+?)\.",
        r"The ([\w\s-]+?) opened only for ([\w\s-]+?) before " + T + r",? and it was the only way into the ([\w\s-]+?)\.",
    ]
    spec = None
    for i, p in enumerate(pats):
        m = re.search(p, ev)
        if m:
            g = m.groups()
            spec = [(g[0], g[1], g[2], g[3]), (g[2], g[0], g[3], g[1]), (g[2], g[0], g[1], g[3])][i]
            break
    if not spec:
        return []
    t0, door, group, area = tmin(spec[0]), spec[1], spec[2], spec[3]
    # exclusive groups: 'nobody on the volunteer rota is on the safety-trained list' / 'no visitor is a lifeguard'
    excl = re.findall(r"[Nn]obody (?:on the |who is (?:a |an )?)?([\w\s-]+?) is (?:on the |a |an )?[\w\s-]+?[.,]", ev)
    excl += re.findall(r"\b[Nn]o ([\w\s-]+?) (?:is|was) (?:a |an |on the )?[\w\s-]+?[.,]", ev)
    if not excl:
        return []
    nonmembers = set()
    for clause in re.split(r"(?<=\.) |, and ", ev):
        if re.search(r"\b([Nn]obody|[Nn]o)\b", clause):
            continue
        for x in excl:
            xw = x.strip().split()[-1]
            if re.search(r"\b(is|are) (?:on the |a |an )?[\w\s-]*" + re.escape(xw), clause):
                nonmembers |= set(re.findall(NAME, clause)) - STOPCAPS
    names = all_names(ev, spk)
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if area.lower().split()[-1] not in tl or NEG.search(t) or not ENTER.search(t) or "by the" in tl:
            continue
        who = subject(t, spk.get(s), names)
        x = single_time(t)
        if who in nonmembers and x is not None and x < t0:
            flags.append(k)
    return flags


def m_capacity_list(ev, msgs, spk):
    m = re.search(r"carried at most (\w+) passengers\. ([A-Z][\w, ]+?) were all on it\.", ev)
    if not m:
        return []
    listed = set(re.findall(NAME, m[2])) - STOPCAPS
    if len(listed) != num(m[1]):
        return []  # list not full: others could be aboard
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if NEG.search(t) or t.endswith("?") or "waiting" in tl or " on the " not in tl and "aboard" not in tl:
            continue
        if not re.search(r"\b(minibus|bus|boat|ferry|coach|van|train|shuttle|car|taxi)\b", tl):
            continue
        named = [n for n in re.findall(NAME, t) if n not in STOPCAPS and n not in ("AM", "PM")]
        seen = re.search(r"\bsaw\b", tl)
        if seen and len(named) == 1:
            if named[0] not in listed:
                flags.append(k)
        elif not named and re.search(r"\bI\b", t) and not seen:
            if spk.get(s) not in listed:
                flags.append(k)
    return flags


def _left_claims(msgs, spk, exits_by):
    """'I didn't leave the X at any point between a and b' by P with an exit in [a, b]."""
    flags = []
    for k, s, t in msgs:
        if not re.search(r"(didn't|did not|never) leave|did I leave", t):
            continue
        ts = times(t)
        if len(ts) != 2 or spk.get(s) not in exits_by:
            continue
        a, b = sorted(ts)
        if any(a <= x <= b for x in exits_by[spk[s]]):
            flags.append(k)
    return flags


def m_left_back(ev, msgs, spk):
    exits = {}
    for p, a, _b in re.findall(NAME + r" left the [\w\s-]+? at " + T + r",? (?:and|then) came back at " + T, ev):
        exits.setdefault(p, []).append(tmin(a))
    log = re.search(r"exit log accurately records every time someone left the [\w\s-]+? that day\. It shows exactly \w+ exits?: ([^.]+)\.", ev)
    if log:
        for p, a in re.findall(NAME + " at " + T, log[1]):
            exits.setdefault(p, []).append(tmin(a))
    return _left_claims(msgs, spk, exits) if exits else []


def m_stayed(ev, msgs, spk):
    flags = []
    names = all_names(ev, spk)
    for p, r1, a, b in re.findall(NAME + r" stayed in the ([\w\s-]+?) from " + T + " to " + T, ev):
        sep = re.search(r"[Tt]he ([\w\s-]+?) and the ([\w\s-]+?) are separate (?:rooms|locations)", ev)
        if not sep:
            continue
        others = {sep[1], sep[2]} - {r1}
        if len(others) != 1:
            continue
        r2 = others.pop()
        lo, hi = tmin(a), tmin(b)
        for k, s, t in msgs:
            tl = t.lower()
            if "in the " + r2.lower() not in tl or NEG.search(t):
                continue
            who = subject(t, spk.get(s), names)
            x = single_time(t)
            if who == p and x is not None and lo <= x <= hi:
                flags.append(k)
    return flags


def m_weekday_ban(ev, msgs, spk):
    wdf = _weekday_fn(ev)
    m = re.search(r"[Nn]obody could be inside the ([\w\s-]+?) on any (\w+day) in " + MONTH_RE, ev) or \
        re.search(r"On any (\w+day) in " + MONTH_RE + r" \d{4},? nobody could be inside the ([\w\s-]+?)\.", ev)
    if not (wdf and m):
        return []
    if m.re.pattern.startswith("On"):
        wd, place = WEEKDAYS.index(m[1]), m[3]
    else:
        place, wd = m[1], WEEKDAYS.index(m[2])
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if "inside the " + place.lower() not in tl or NEG.search(t) or "outside" in tl:
            continue
        ds = dates(t)
        if len(ds) == 1 and wdf(ds[0]) == wd:
            flags.append(k)
    return flags


def m_sealed(ev, msgs, spk):
    m = re.search(r"The ([\w\s-]+?) stayed sealed from " + T + " to " + T + r", and the ([\w\s-]+?) could not be removed without breaking the seal", ev)
    m2 = re.search(r"From " + T + " to " + T + r",? the ([\w\s-]+?) stayed sealed, and the ([\w\s-]+?) could not be removed without breaking the seal", ev)
    if m:
        item, lo, hi = m[4], tmin(m[2]), tmin(m[3])
    elif m2:
        item, lo, hi = m2[4], tmin(m2[1]), tmin(m2[2])
    else:
        return []
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if item.lower() not in tl or NEG.search(t) or not re.search(r"\b(out of|removed|took|pulled|taken)\b", tl):
            continue
        x = single_time(t)
        if x is not None and lo <= x <= hi:
            flags.append(k)
    return flags


def m_holder_durations(ev, msgs, spk):
    """'From T P held it for exactly D, then handed it to Q. Q kept it for exactly D, then handed it to R, who kept it for exactly D.'"""
    w = re.search(r"(?:[Bb]etween " + T + " and " + T + r",? nobody else held the ([\w\s-]+?)[,.]|"
                  r"[Nn]obody else held the ([\w\s-]+?) between " + T + " and " + T + ")", ev)
    st = re.search(r"From " + T + r",? " + NAME + r" (?:held|kept) it for exactly " + DUR, ev)
    if not (w and st) or "only one person held" not in ev.lower():
        return []
    ws, we, item = (w[1], w[2], w[3]) if w[1] else (w[5], w[6], w[4])
    ws, we = tmin(ws), tmin(we)
    seq = [(st[2], dur(st[3]))]
    rest = ev[st.end():]
    for m in re.finditer(r"handed it to " + NAME + r"(?:\. " + NAME + r"|, who)? (?:kept|held) it for exactly " + DUR, rest):
        seq.append((m[1], dur(m[3])))
    bounds, cur = [], tmin(st[1])
    for p, d in seq:
        bounds.append((p, cur, cur + d))
        cur += d
    if cur != we or tmin(st[1]) != ws or len(seq) != len(re.findall(r"for exactly", ev)):
        return []
    names = all_names(ev, spk)
    key = item.lower().split()[-1]
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if key not in tl or NEG.search(t) or t.endswith("?") or re.search(r"\b(put|gave|handed|left|log)\b", tl):
            continue
        if not re.search(r"\b(held|holding|had|hands|kept)\b", tl):
            continue
        who = subject(t, spk.get(s), names)
        x = single_time(t)
        if who and x is not None:
            h = [p for p, a, b in bounds if a < x < b]
            if h and h[0] != who:
                flags.append(k)
    return flags


def m_entry_log(ev, msgs, spk):
    m = re.search(r"(?:log|register) accurately records every entry to the ([\w\s-]+?) that day[^:]*?(?:shows|lists) exactly \w+ entr(?:y|ies): ([^.]+)\.", ev)
    if not m:
        return []
    place = m[1]
    entries = {(p, tmin(x)) for p, x in re.findall(NAME + " at " + T, m[2])}
    flags = []
    for k, s, t in msgs:
        tl = t.lower()
        if place.lower() not in tl or NEG.search(t) or not re.search(r"\b(entered|went into|got into|stepped into|walked into|came into)\b", tl):
            continue
        if [n for n in re.findall(NAME, t) if n not in STOPCAPS and n not in ("AM", "PM")]:
            continue
        x = single_time(t)
        if x is not None and (spk.get(s), x) not in entries:
            flags.append(k)
    return flags


MODULES += [m_gate_group, m_capacity_list, m_left_back, m_stayed, m_weekday_ban, m_sealed,
            m_holder_durations, m_entry_log]


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


