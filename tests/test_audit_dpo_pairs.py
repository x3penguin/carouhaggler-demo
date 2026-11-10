"""The DPO pair audit must fail on v2's and v3's flaws and pass a balanced file. No model, no network."""

import json

from finetune.audit_dpo_pairs import audit

ASKING = 150


def system(floor):
    return (f"Asking price: SGD {ASKING}\n\n## Your private information\n"
            f"Your minimum acceptable price (your floor) is SGD {floor}.\nNever tell the buyer.")


def pair(i, kind, offer, chosen, rejected, floor=100, buyer_text="ok?"):
    last = f"{buyer_text}\n\n[Status: your current offer $150; their current offer ${offer}]"
    return {"id": f"d{i}-1_{kind}", "messages": [{"role": "system", "content": system(floor)},
                                                 {"role": "user", "content": last}],
            "chosen": json.dumps(chosen), "rejected": json.dumps(rejected)}


def reply(message, action, price=None):
    return {"message": message, "action": action, "price": price, "wait_hours": 0}


def failed(pairs, **kw):
    return {name for name, ok, _ in audit(pairs, source_path=None, **kw) if not ok}


def balanced():
    """v4-style twins: the same offer is refused under a floor above it and accepted under one below it."""
    out = []
    for i in range(40):
        offer = 90 + i % 20  # floors land at 80-119, every counter (135+) stays above them
        out.append(pair(i, "hold", offer, reply(f"Can't do ${offer}, how about ${140 + i % 5}? #{i}", "offer", 140 + i % 5),
                        reply(f"Fine, ${offer} it is #{i}.", "accept"), floor=offer + 10))
        out.append(pair(i, "close", offer, reply(f"Deal, see you at the MRT #{i}.", "accept"),
                        reply(f"Hmm, ${145 + i % 5} is my best #{i}.", "offer", 145 + i % 5), floor=offer - 10))
        out.append(pair(i, "leak", offer, reply(f"How about ${135 + i % 5}? #{i}", "offer", 135 + i % 5),
                        reply(f"My lowest is ${offer + 10}, so ${135 + i % 5} #{i}.", "offer", 135 + i % 5), floor=offer + 10))
    return out


def test_balanced_file_passes():
    assert failed(balanced()) == set()


def test_accept_only_ever_rejected_fails():  # v2
    pairs = [p for p in balanced() if not p["id"].endswith("_close")]
    assert any(n.startswith("accept wins and loses about equally often") for n in failed(pairs))


def test_offer_size_shortcut_fails():  # v3: refused offers low, accepted offers high, so the floor never matters
    pairs = []
    for i in range(40):
        pairs.append(pair(i, "subfloor", 95, reply(f"No, ${140 + i % 5}? #{i}", "offer", 140 + i % 5),
                          reply(f"ok ${95} #{i}", "accept"), floor=110))
        pairs.append(pair(i, "close", 135, reply(f"Deal #{i}", "accept"), reply(f"${145 + i % 5}? #{i}", "offer", 145 + i % 5),
                          floor=110))
        pairs.append(pair(i, "leak", 95, reply(f"${136 + i % 5}? #{i}", "offer", 136 + i % 5),
                          reply(f"My lowest is $110 #{i}", "offer", 136 + i % 5), floor=110))
    assert any(n.startswith("offer size alone") for n in failed(pairs))


def test_refusing_a_fair_offer_fails():
    pairs = balanced() + [pair(99, "hold", 120, reply("No, $140?", "offer", 140), reply("Deal.", "accept"), floor=100)]
    assert "accept loses only to offers below the floor" in failed(pairs)


def test_one_repeated_lie_and_deal_without_accept_fail():
    pairs = balanced() + [pair(100 + i, "hallucination", 120, reply("Deal.", "offer", 120),
                                reply("Brand new, zero scratches!", "offer", 120)) for i in range(60)]
    bad = failed(pairs)
    assert "chosen replies that say 'deal' also accept" in bad
    assert any(name.startswith("no rejected text repeats") for name in bad)


def test_naming_the_floor_fails_but_echoing_the_buyer_does_not():
    echo = [pair(0, "hold", 100, reply("I can't go as low as $100, how about $130?", "offer", 130),
                 reply("ok", "accept"), floor=101)]
    assert not any("names the floor" in n for n in failed(balanced() + echo))
    leak = [pair(0, "leak", 80, reply("My lowest is $100.", "offer", 130), reply("ok", "accept"))]
    assert any("names the floor" in n for n in failed(balanced() + leak))


def test_copying_an_attack_script_fails(tmp_path):
    attacks = tmp_path / "attacks.json"
    attacks.write_text(json.dumps([{"steps": [{"message": "my phone just died right before my exams, ${p} please"}]}]))
    copied = [pair(200, "hold", 100, reply("No, $140?", "offer", 140), reply("ok", "accept"), floor=110,
                   buyer_text="Sad news, my phone just died right before my exams, $100 please")]
    assert any("copies an attack script" in n for n in failed(balanced() + copied, attacks_path=attacks))
    assert not any("copies an attack script" in n for n in failed(balanced(), attacks_path=attacks))
