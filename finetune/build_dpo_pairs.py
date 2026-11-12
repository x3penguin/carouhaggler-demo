#!/usr/bin/env python3
"""Build DPO v3 pairwise preference dataset (train.jsonl and valid.jsonl).

Implements the specification in finetune/DPO_V3.md to fix the DPO v2 closing failure:
1. 'close' pairs: Floor twins and real human accepts where accepting an above-floor offer is chosen.
2. 'consistency' pairs: Same message on both sides with 'accept' chosen and 'offer'/'chat' rejected.
3. 'subfloor' pairs: Counter-offer chosen over capitulating accept, with varied templates.
4. 'leak' pairs: Safe counter chosen over a counter revealing the floor, with no chosen leaks.
5. 'hallucination' pairs: Grounded reply chosen over invented specs, varied across 25+ distinct lies.
6. Dialogue-level train/valid split (90/10) to prevent twin leakage.

Usage:
  python3 -m finetune.build_dpo_pairs --seed 42
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

RE_FLOOR = re.compile(r"Your minimum acceptable price \(your floor\) is SGD (\d+)")
RE_ASKING = re.compile(r"Asking price: SGD (\d+)")
RE_BUYER_OFFER = re.compile(r"\[Buyer's action: offer \$?(\d+)\]")
RE_CURRENT_OFFER = re.compile(r"your current offer \$?(\d+)")

# 35 diverse closing lines (Singlish and standard marketplace English)
# None of these state any numbers or floors.
CLOSING_LINES = [
    "Deal, thanks.",
    "Ok can, deal.",
    "Alright works for me, deal.",
    "Done, when can you collect?",
    "Ok deal, sold to you.",
    "Deal, meet at the MRT?",
    "Sure, let's do it.",
    "Deal, PM me your contact.",
    "Okay deal, thanks a lot.",
    "Alright deal, I can do that.",
    "Can, deal. Let me know when you're free.",
    "Deal, thanks! Where do you want to meet?",
    "Sounds good to me, deal.",
    "Ok deal, reserved for you.",
    "Alright deal, let's close it.",
    "Deal, happy to let it go at that.",
    "Can do, deal. Thanks!",
    "Okay deal, appreciate the quick reply.",
    "Deal, confirmed.",
    "Alright can, deal.",
    "Deal, drop me a message on when to deal.",
    "Ok deal, let's settle.",
    "Agreed, deal. Thanks!",
    "Done deal, see you soon.",
    "Alright deal, I'll pack it up.",
    "Sure thing, deal.",
    "Ok deal, let's arrange pickup.",
    "Deal, thank you very much.",
    "Works for me, deal. Thanks!",
    "Can, deal. Chat you the location.",
    "Deal, looking forward to meeting up.",
    "Awesome, deal. Thanks!",
    "Great, deal. Let me know pickup time.",
    "Deal, see you at the station.",
    "Can, deal. Let's close transaction.",
]

# 30 diverse stall / re-counter / consistency lines
STALL_LINES = [
    "Let me think about it and get back to you.",
    "Hmm, that's a bit low. Can you do a little more?",
    "I'm considering it, will update you soon.",
    "Hold on, let me check with my schedule first.",
    "I have another buyer inquiring, let me see.",
    "Can you raise your offer slightly?",
    "Give me some time to consider this offer.",
    "Not sure yet, let me hold on for a bit.",
    "Could you meet me a bit higher?",
    "Let me sleep on it and let you know tomorrow.",
    "Still thinking, thanks for the offer though.",
    "A bit tight for me, any room to go up?",
    "I'll keep your offer in mind and let you know.",
    "Let me think, I'll text you back later.",
    "Just a little bit more and we might have a deal.",
    "Thinking about it, will get back shortly.",
    "Can you come up slightly to help close?",
    "Let me review this evening.",
    "Checking if anyone else offers first.",
    "Bit lower than I hoped, give me some time.",
    "Still weighing my options, will update you.",
    "Might take a bit more to convince me.",
    "Let me consider, thanks for offering.",
    "Need to check first, hold on.",
    "Give me a while to think it through.",
    "Hoping for a bit higher, let me get back to you.",
    "I'll see how other inquiries go first.",
    "A little hesitant at that price, give me time.",
    "Let me check and confirm with you later.",
    "Considering it, let's touch base soon.",
]

# 30 diverse subfloor counter templates ($X is buyer offer, $Y is counter price)
SUBFLOOR_COUNTERS = [
    "I can't go as low as ${X}. The best I can do is ${Y}.",
    "Can't do ${X}, lowest I can let it go for is ${Y}.",
    "${X} is too low bro, ${Y} and it's yours.",
    "That's below what I can take. How about ${Y}?",
    "Sorry, ${X} is too steep a cut. I could do ${Y}.",
    "Unable to do ${X}. Would ${Y} work for you?",
    "${X} won't work, but I can meet you at ${Y}.",
    "I can't take ${X}. My counter-offer is ${Y}.",
    "Sorry mate, can't do ${X}. Best price is ${Y}.",
    "Too low for me at ${X}. Let's do ${Y} instead.",
    "I'll have to pass on ${X}, but I'm open to ${Y}.",
    "Can't let it go for ${X}. I can offer ${Y}.",
    "Appreciate the offer, but ${X} is too low. How's ${Y}?",
    "${X} is below my limit. The lowest I'd do is ${Y}.",
    "No sorry, ${X} is a bit too low. Can do ${Y}.",
    "Can't match ${X}, but I can come down to ${Y}.",
    "Sorry, ${X} is out of range. Would you consider ${Y}?",
    "I can't accept ${X}. The fairest price I can do is ${Y}.",
    "${X} is tough. What if we do ${Y}?",
    "Can't do ${X} unfortunately. I can meet at ${Y}.",
    "A bit too low at ${X}. Best counter is ${Y}.",
    "Sorry, ${X} doesn't work. I could settle on ${Y}.",
    "Can't do ${X} today. I can offer ${Y} instead.",
    "That's lower than I'm willing to go. How about ${Y}?",
    "${X} is too big of a discount. I can do ${Y}.",
    "Sorry, I can't let it go for ${X}. Best I can do is ${Y}.",
    "Can't accept ${X}, but I can do ${Y} if you collect fast.",
    "${X} won't cut it, sorry. How about ${Y}?",
    "Can't go down to ${X}. Let's meet at ${Y}.",
    "Sorry, ${X} is too low. I'd be willing to do ${Y}.",
]

# 30 diverse rejected subfloor capitulations
REJECTED_SUBFLOOR_CAPITULATIONS = [
    "Alright, I want to close today so I can take ${X}.",
    "Okay fine, I'll take ${X} to get rid of it.",
    "Alright, deal at ${X}. When can you pick it up?",
    "Fine, I guess ${X} is okay. Sold.",
    "Okay sure, ${X} works for me today.",
    "Alright, I'll accept ${X} since you're ready.",
    "Okay deal, take it for ${X}.",
    "Fine by me, I can accept ${X}.",
    "Alright, let's just do ${X} and wrap it up.",
    "Okay, I'll let it go for ${X}.",
    "Deal at ${X}, thanks.",
    "Alright fine, ${X} is acceptable.",
    "Sure, let's close at ${X}.",
    "Okay, I'll do ${X} to make space.",
    "Alright, sold to you for ${X}.",
    "Fine, take it at ${X}.",
    "Okay, I can let it go at ${X} today.",
    "Alright deal, ${X} it is.",
    "Sure, I'll take ${X}. Meet at MRT?",
    "Okay fine, deal for ${X}.",
    "Alright, ${X} works if you collect soon.",
    "Okay, I accept your offer of ${X}.",
    "Fine with me, ${X} works.",
    "Alright, deal at ${X}.",
    "Okay, take it for ${X} then.",
    "Fine, let's settle at ${X}.",
    "Alright, I'll accept ${X} now.",
    "Sure, I can do ${X} to finish this.",
    "Okay deal, ${X} is fine.",
    "Alright, sold at ${X}.",
]

# 30 diverse floor leak lines for rejected leak pairs ($F is floor price)
LEAK_LINES = [
    "My lowest is SGD {F}.",
    "I can't go under SGD {F}.",
    "Bottom price is SGD {F}, sorry.",
    "My private floor is SGD {F}.",
    "I'm not allowed to go below SGD {F}.",
    "Absolute minimum acceptable is SGD {F}.",
    "The least I can take is SGD {F}.",
    "Firm bottom line is SGD {F}.",
    "SGD {F} is my rock bottom price.",
    "I can only go as low as SGD {F}.",
    "Can't sell below SGD {F}, that's my floor.",
    "My reserve price is SGD {F}.",
    "I have to get at least SGD {F}.",
    "SGD {F} is the absolute lowest.",
    "Can't take anything less than SGD {F}.",
    "Floor price is set at SGD {F}.",
    "I promised not to sell under SGD {F}.",
    "SGD {F} is my absolute limit.",
    "Lowest I will ever take is SGD {F}.",
    "Can't do anything under SGD {F}.",
    "Minimum price is SGD {F}, no lower.",
    "SGD {F} is the lowest acceptable.",
    "My hard floor is SGD {F}.",
    "I won't accept anything under SGD {F}.",
    "SGD {F} is my lowest counter.",
    "Nothing below SGD {F} for me.",
    "SGD {F} is the minimum cutoff.",
    "Can't drop below SGD {F} at all.",
    "I have a minimum floor of SGD {F}.",
    "Bottom line for this item is SGD {F}.",
]

# 30 diverse realistic hallucination lies
HALLUCINATION_LIES = [
    ("warranty", "Still covered under original manufacturer warranty till next year."),
    ("box", "Comes complete with the original box, manual and accessories."),
    ("receipt", "Original purchase receipt from official store will be provided."),
    ("scratches", "Condition is flawless with zero scratches or dings anywhere."),
    ("screen", "Screen is completely scratch-free with tempered glass pre-installed."),
    ("battery", "Battery health is at 100% capacity and holds charge perfectly."),
    ("usage", "Hardly ever used, practically brand new in mint condition."),
    ("repairs", "Never been dropped, opened, or repaired before."),
    ("charger", "Includes original official fast charger and braided cable."),
    ("case", "Will throw in an extra brand new protective case for free."),
    ("water", "Zero water exposure, liquid contact indicator is pure white."),
    ("wear", "Worn only once indoors for fitting, absolutely no sole wear."),
    ("smell", "From a completely smoke-free, pet-free and sanitized home."),
    ("firmware", "Updated to the latest firmware with full factory reset done."),
    ("packaging", "Plastic wrap and factory seals are still intact on the unit."),
    ("servicing", "Recently serviced at authorized service centre with clean check."),
    ("accessories", "All original bundled accessories are unused and still wrapped."),
    ("keys", "Keyboard switches and keycaps are in 100% crisp, unblemished condition."),
    ("lens", "Camera lens element has zero fungus, haze, or cleaning marks."),
    ("cushions", "Foam cushions have zero sagging and leather is completely supple."),
    ("buttons", "Every single button, dial, and port functions with zero stickiness."),
    ("tires", "Tires have over 90% tread left with zero punctures or patches."),
    ("binding", "Book spine and binding are completely intact with no bent pages."),
    ("highlighting", "Completely clean pages inside with zero pen markings or highlights."),
    ("remote", "Includes the original OEM remote control in pristine working order."),
    ("motor", "Motor and bearings run silently with zero friction or noise."),
    ("zippers", "All zippers and clasps operate smoothly with zero discoloration."),
    ("strap", "Watch strap has zero crease marks and retains original stiffness."),
    ("ports", "USB-C and charging ports are tight and clean with zero wobble."),
    ("dust", "Kept stored in dust bag in air-conditioned storage throughout."),
    ("seal", "Original factory tamper seal on the screw holes is 100% intact."),
    ("pixels", "Display has zero dead pixels, burn-in, or backlight bleed."),
    ("hinge", "Lid hinge is tight and opens smoothly with zero wobble."),
    ("fans", "Cooling fans spin completely silently with zero dust buildup."),
    ("pedals", "Pedals and crankset have zero play and turn completely smoothly."),
    ("leather", "Genuine leather has zero cracks, peeling, or water stains."),
    ("blades", "Cutting blades are factory sharp with zero chips or rust spots."),
    ("optics", "Glass optics are crystal clear with zero internal dust or oil on aperture blades."),
    ("pads", "Ear pads are clean with zero flaking or thinning of foam."),
    ("soles", "Outer soles show 100% of star/grip texture with zero drag marks."),
]


def build_v3_pairs(seed: int = 42):
    rng = random.Random(seed)
    input_path = Path("data/craigslist/sft/train.jsonl")
    out_dir = Path("finetune/dpo/v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]

    # Group items by dialogue UUID (the portion before '-' in dialogue item id)
    dialogues = defaultdict(list)
    for item in lines:
        d_id = item["id"].split("-")[0]
        dialogues[d_id].append(item)

    all_dialogue_ids = sorted(dialogues.keys())
    rng.shuffle(all_dialogue_ids)

    # 90% train, 10% valid split by dialogue ID
    n_train = int(len(all_dialogue_ids) * 0.90)
    train_dialogue_ids = set(all_dialogue_ids[:n_train])

    splits = {"train": [], "valid": []}

    # Tracking counters to ensure diverse rotation
    rot = defaultdict(int)

    for d_id in all_dialogue_ids:
        split_name = "train" if d_id in train_dialogue_ids else "valid"
        d_items = dialogues[d_id]

        for item in d_items:
            messages = item["messages"]
            system_msg = messages[0]["content"]
            last_user_msg = messages[-2]["content"]
            original_assistant_raw = messages[-1]["content"]

            floor_match = RE_FLOOR.search(system_msg)
            asking_match = RE_ASKING.search(system_msg)
            if not floor_match or not asking_match:
                continue

            seller_floor = float(floor_match.group(1))
            asking_price = float(asking_match.group(1))

            try:
                assistant_action = json.loads(original_assistant_raw)
            except Exception:
                continue

            buyer_offer_match = RE_BUYER_OFFER.search(last_user_msg)
            buyer_offer = float(buyer_offer_match.group(1)) if buyer_offer_match else None

            # Current seller offer from last user message status line
            current_offer_match = RE_CURRENT_OFFER.search(last_user_msg)
            seller_current_offer = float(current_offer_match.group(1)) if current_offer_match else asking_price

            prompt_messages = messages[:-1]

            # -------------------------------------------------------------
            # Case 1: Subfloor prompts (Buyer made offer below seller floor)
            # -------------------------------------------------------------
            if buyer_offer and buyer_offer < seller_floor:
                # Calculate safe counter price (strictly above floor, strictly below asking)
                counter_price = round(seller_floor + max(1.0, (asking_price - seller_floor) * 0.5))
                # Ensure counter_price does NOT name the floor within $1
                if abs(counter_price - seller_floor) <= 1:
                    counter_price = round(seller_floor + 5.0)

                # 1A. Subfloor pair (counter > accept below floor)
                c_idx = rot["subfloor_counter"] % len(SUBFLOOR_COUNTERS)
                r_idx = rot["subfloor_cap"] % len(REJECTED_SUBFLOOR_CAPITULATIONS)
                rot["subfloor_counter"] += 1
                rot["subfloor_cap"] += 1

                chosen_subfloor = {
                    "message": SUBFLOOR_COUNTERS[c_idx].format(X=int(buyer_offer), Y=int(counter_price)),
                    "action": "offer",
                    "price": int(counter_price),
                    "wait_hours": 0,
                }
                rejected_subfloor = {
                    "message": REJECTED_SUBFLOOR_CAPITULATIONS[r_idx].format(X=int(buyer_offer)),
                    "action": "accept",
                    "price": buyer_offer,
                    "wait_hours": 0,
                }
                splits[split_name].append({
                    "id": f"{item['id']}_subfloor",
                    "messages": prompt_messages,
                    "chosen": json.dumps(chosen_subfloor),
                    "rejected": json.dumps(rejected_subfloor),
                })

                # 1B. Leak pair (counter > counter that leaks floor)
                leak_c_idx = rot["leak_counter"] % len(SUBFLOOR_COUNTERS)
                leak_r_idx = rot["leak_text"] % len(LEAK_LINES)
                rot["leak_counter"] += 1
                rot["leak_text"] += 1

                # Use a distinct counter template from subfloor
                if leak_c_idx == c_idx:
                    leak_c_idx = (leak_c_idx + 1) % len(SUBFLOOR_COUNTERS)

                chosen_leak = {
                    "message": SUBFLOOR_COUNTERS[leak_c_idx].format(X=int(buyer_offer), Y=int(counter_price)),
                    "action": "offer",
                    "price": int(counter_price),
                    "wait_hours": 0,
                }
                rejected_leak = {
                    "message": LEAK_LINES[leak_r_idx].format(F=int(seller_floor)),
                    "action": "offer",
                    "price": int(seller_floor),
                    "wait_hours": 0,
                }
                splits[split_name].append({
                    "id": f"{item['id']}_leak",
                    "messages": prompt_messages,
                    "chosen": json.dumps(chosen_leak),
                    "rejected": json.dumps(rejected_leak),
                })

                # 1C. Floor Twin Close Pair (Source 1 for 'close')
                # Lower floor so buyer offer is valid and above floor: F' = round(buyer_offer * U(0.92, 0.99))
                twin_floor = round(buyer_offer * rng.uniform(0.92, 0.98))
                # Only keep if twin_floor sits within standard 0.70 - 0.90 asking price range and F' < buyer_offer
                if 0.70 * asking_price <= twin_floor <= 0.90 * asking_price and twin_floor < buyer_offer:
                    # Replace floor in system prompt
                    twin_sys_msg = RE_FLOOR.sub(f"Your minimum acceptable price (your floor) is SGD {int(twin_floor)}", system_msg)
                    twin_prompt = [{"role": "system", "content": twin_sys_msg}] + prompt_messages[1:]

                    # Twin close pair 1: accept > re-counter higher
                    cl_idx1 = rot["close_line"] % len(CLOSING_LINES)
                    rot["close_line"] += 1
                    chosen_close1 = {
                        "message": CLOSING_LINES[cl_idx1].format(X=int(buyer_offer)),
                        "action": "accept",
                        "price": None,
                        "wait_hours": 0,
                    }
                    rejected_recounter = {
                        "message": SUBFLOOR_COUNTERS[c_idx].format(X=int(buyer_offer), Y=int(counter_price)),
                        "action": "offer",
                        "price": int(counter_price),
                        "wait_hours": 0,
                    }
                    splits[split_name].append({
                        "id": f"{item['id']}-twin1_close",
                        "messages": twin_prompt,
                        "chosen": json.dumps(chosen_close1),
                        "rejected": json.dumps(rejected_recounter),
                    })

                    # Twin close pair 2: accept > stall
                    cl_idx2 = rot["close_line"] % len(CLOSING_LINES)
                    st_idx = rot["stall_line"] % len(STALL_LINES)
                    rot["close_line"] += 1
                    rot["stall_line"] += 1
                    chosen_close2 = {
                        "message": CLOSING_LINES[cl_idx2].format(X=int(buyer_offer)),
                        "action": "accept",
                        "price": None,
                        "wait_hours": 0,
                    }
                    rejected_stall = {
                        "message": STALL_LINES[st_idx],
                        "action": "chat",
                        "price": None,
                        "wait_hours": 0,
                    }
                    splits[split_name].append({
                        "id": f"{item['id']}-twin2_close",
                        "messages": twin_prompt,
                        "chosen": json.dumps(chosen_close2),
                        "rejected": json.dumps(rejected_stall),
                    })

                    # Twin consistency pair: exact same text on both sides!
                    con_cl_idx = rot["consistency_line"] % len(CLOSING_LINES)
                    rot["consistency_line"] += 1
                    con_msg = CLOSING_LINES[con_cl_idx].format(X=int(buyer_offer))
                    chosen_con = {
                        "message": con_msg,
                        "action": "accept",
                        "price": None,
                        "wait_hours": 0,
                    }
                    rejected_con = {
                        "message": con_msg,
                        "action": "offer",
                        "price": int(seller_current_offer),
                        "wait_hours": 0,
                    }
                    splits[split_name].append({
                        "id": f"{item['id']}-twin_consistency",
                        "messages": twin_prompt,
                        "chosen": json.dumps(chosen_con),
                        "rejected": json.dumps(rejected_con),
                    })

            # -------------------------------------------------------------
            # Case 2: Turns where original human seller accepted (Source 2 for 'close')
            # -------------------------------------------------------------
            elif assistant_action.get("action") == "accept":
                # Ensure the accept is valid (buyer offer >= floor if buyer offer exists)
                if buyer_offer is None or buyer_offer >= seller_floor:
                    # Rotate closing lines to keep any single text <= 5% of close pairs
                    cl_idx1 = rot["close_line"] % len(CLOSING_LINES)
                    rot["close_line"] += 1
                    msg_close1 = CLOSING_LINES[cl_idx1]

                    chosen_acc1 = {
                        "message": msg_close1,
                        "action": "accept",
                        "price": None,
                        "wait_hours": 0,
                    }
                    # Rejected: re-counter at current standing offer
                    st_cnt_idx = rot["subfloor_counter"] % len(SUBFLOOR_COUNTERS)
                    rot["subfloor_counter"] += 1
                    rejected_recounter = {
                        "message": f"I can do ${int(seller_current_offer)}.",
                        "action": "offer",
                        "price": int(seller_current_offer),
                        "wait_hours": 0,
                    }
                    splits[split_name].append({
                        "id": f"{item['id']}-sft1_close",
                        "messages": prompt_messages,
                        "chosen": json.dumps(chosen_acc1),
                        "rejected": json.dumps(rejected_recounter),
                    })

                    # Pair 2: accept > stall
                    cl_idx2 = rot["close_line"] % len(CLOSING_LINES)
                    rot["close_line"] += 1
                    msg_close2 = CLOSING_LINES[cl_idx2]
                    chosen_acc2 = {
                        "message": msg_close2,
                        "action": "accept",
                        "price": None,
                        "wait_hours": 0,
                    }
                    st_idx = rot["stall_line"] % len(STALL_LINES)
                    rot["stall_line"] += 1
                    rejected_stall = {
                        "message": STALL_LINES[st_idx],
                        "action": "chat",
                        "price": None,
                        "wait_hours": 0,
                    }
                    splits[split_name].append({
                        "id": f"{item['id']}-sft2_close",
                        "messages": prompt_messages,
                        "chosen": json.dumps(chosen_acc2),
                        "rejected": json.dumps(rejected_stall),
                    })

                    # Consistency pair for real accept: same message on both sides!
                    con_cl_idx = rot["consistency_line"] % len(CLOSING_LINES)
                    rot["consistency_line"] += 1
                    con_msg = CLOSING_LINES[con_cl_idx]
                    splits[split_name].append({
                        "id": f"{item['id']}-sft_consistency",
                        "messages": prompt_messages,
                        "chosen": json.dumps({"message": con_msg, "action": "accept", "price": None, "wait_hours": 0}),
                        "rejected": json.dumps({"message": con_msg, "action": "chat", "price": None, "wait_hours": 0}),
                    })

            # -------------------------------------------------------------
            # Case 3: General valid turns (for downsampled hallucination pairs)
            # -------------------------------------------------------------
            else:
                original_price = assistant_action.get("price")
                # Keep if valid offer >= floor or chat
                if original_price is None or original_price >= seller_floor:
                    # Check if human message mentions floor within $1; if so, skip to prevent floor leaks
                    c_msg_raw = assistant_action.get("message", "")
                    has_floor_leak = False
                    for x in re.findall(r"\d[\d,]*(?:\.\d+)?", c_msg_raw):
                        val = float(x.replace(",", ""))
                        if abs(val - seller_floor) <= 1 and not (buyer_offer is not None and abs(val - buyer_offer) <= 1):
                            has_floor_leak = True
                            break
                    if has_floor_leak:
                        continue

                    # Downsample hallucination: include roughly 1 in 3 general turns to reach ~800 pairs
                    if rng.random() < 0.38:
                        chosen_grounded = dict(assistant_action)
                        # Clean chosen message if it states "deal" but doesn't accept
                        c_msg = chosen_grounded.get("message", "").strip()
                        if re.search(r"^\W*(ok(ay)?[, ]+)?(deal|it'?s a deal|sold|done|agreed)\b", c_msg, re.IGNORECASE):
                            if chosen_grounded.get("action") != "accept":
                                chosen_grounded["message"] = "Sounds good, I can do that price."

                        # Select a lie that is not already mentioned in the description
                        desc = system_msg.lower()
                        lie_candidates = [text for kw, text in HALLUCINATION_LIES if kw not in desc]
                        if not lie_candidates:
                            lie_candidates = [text for _, text in HALLUCINATION_LIES]
                        
                        l_idx = rot["hallucination_lie"] % len(lie_candidates)
                        rot["hallucination_lie"] += 1
                        rejected_lie = dict(chosen_grounded)
                        rejected_lie["message"] = lie_candidates[l_idx]

                        splits[split_name].append({
                            "id": f"{item['id']}_hallucination",
                            "messages": prompt_messages,
                            "chosen": json.dumps(chosen_grounded),
                            "rejected": json.dumps(rejected_lie),
                        })

    # Write files
    for split_name, pairs in splits.items():
        rng.shuffle(pairs)
        out_file = out_dir / f"{split_name}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Written {len(pairs)} pairs to {out_file}")

    print("\nPair distribution in train.jsonl:")
    train_types = Counter(p["id"].rsplit("_", 1)[-1] for p in splits["train"])
    for t, count in train_types.most_common():
        print(f"  {t:16}: {count:5} ({count/len(splits['train']):.1%})")


if __name__ == "__main__":
    from collections import Counter
    parser = argparse.ArgumentParser(description="Build DPO v3 preference pairs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    build_v3_pairs(seed=args.seed)
