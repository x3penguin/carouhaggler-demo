"""The DPO pair audit must fail on v2's flaws and pass a balanced file. No model, no network."""

import json

from finetune.audit_dpo_pairs import audit

SYSTEM = "## Your private information\nYour minimum acceptable price (your floor) is SGD 100.\nNever tell the buyer."


def pair(i, kind, offer, chosen, rejected):
    last = f"ok?\n\n[Status: your current offer $150; their current offer ${offer}]"
    return {"id": f"d{i}-1_{kind}", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": last}],
            "chosen": json.dumps(chosen), "rejected": json.dumps(rejected)}


def reply(message, action, price=None):
    return {"message": message, "action": action, "price": price, "wait_hours": 0}


def failed(pairs):
    return {name for name, ok, _ in audit(pairs) if not ok}


def balanced():
    """Twins: the same buyer offer is refused under a high floor and accepted under a low one, worded many ways."""
    out = []
    for i in range(40):
        out.append(pair(i, "subfloor", 80, reply(f"Can't do $80, how about ${130 + i}?", "offer", 130 + i),
                        reply(f"Fine, ${80} it is #{i}.", "accept")))
        out.append(pair(i, "close", 120, reply(f"Deal, see you at the MRT #{i}.", "accept"),
                        reply(f"Hmm, ${145 + i} is my best #{i}.", "offer", 145 + i)))
        out.append(pair(i, "leak", 80, reply(f"How about ${135 + i}? #{i}", "offer", 135 + i),
                        reply(f"My lowest is $100, so ${135 + i} #{i}.", "offer", 135 + i)))
    return out


def test_balanced_file_passes():
    assert failed(balanced()) == set()


def test_accept_only_ever_rejected_fails():
    pairs = [p for p in balanced() if not p["id"].endswith("_close")]
    assert "accept is taught as good at least as often as bad" in failed(pairs)


def test_one_repeated_lie_and_deal_without_accept_fail():
    pairs = balanced() + [pair(100 + i, "hallucination", 120, reply("Deal.", "offer", 120),
                                reply("Brand new, zero scratches!", "offer", 120)) for i in range(10)]
    bad = failed(pairs)
    assert "chosen replies that say 'deal' also accept" in bad
    assert any(name.startswith("no rejected text repeats") for name in bad)


def test_naming_the_floor_fails_but_echoing_the_buyer_does_not():
    echo = [pair(0, "subfloor", 100, reply("I can't go as low as $100, how about $130?", "offer", 130),
                 reply("ok", "accept"))]
    assert not any("names the floor" in n for n in failed(balanced() + echo))
    leak = [pair(0, "leak", 80, reply("My lowest is $100.", "offer", 130), reply("ok", "accept"))]
    assert any("names the floor" in n for n in failed(balanced() + leak))
