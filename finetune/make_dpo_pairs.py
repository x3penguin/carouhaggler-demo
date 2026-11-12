"""Generate DPO pairwise preference dataset from Craigslist SFT training data.

This script parses data/craigslist/sft/train.jsonl and constructs preference pairs:
- 'chosen' (y_w): Responses that respect price floors, negotiate constructively, and maintain grounding.
- 'rejected' (y_l): Responses that commit the 3 core failure modes (conceding below floor,
                   leaking the floor price, or hallucinating item condition).
"""

import json
import re
from pathlib import Path

INPUT_PATH = Path("data/craigslist/sft/train.jsonl")
OUTPUT_DIR = Path("finetune/dpo/v2")
OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"

RE_FLOOR = re.compile(r"Your minimum acceptable price \(your floor\) is SGD (\d+)")
RE_ASKING = re.compile(r"Asking price: SGD (\d+)")
RE_BUYER_OFFER = re.compile(r"\[Buyer's action: offer \$?(\d+)\]")

def make_dpo_pairs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [json.loads(line) for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()]
    
    pairs = []
    
    for item in lines:
        messages = item["messages"]
        system_msg = messages[0]["content"]
        last_user_msg = messages[-2]["content"]
        original_assistant_raw = messages[-1]["content"]
        
        # Extract financial boundaries from system prompt
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
        
        # Build prompt: all messages up to the user message
        prompt_messages = messages[:-1]
        
        # Scenario 1: Buyer made an offer below the seller's floor
        if buyer_offer and buyer_offer < seller_floor:
            # Desired counter-offer price (midpoint between floor and asking, rounded)
            counter_target = round((seller_floor + asking_price) / 2.0)
            
            # CHOSEN: Firm refusal of sub-floor offer, counters safely above floor
            chosen = {
                "message": f"I can't go as low as ${int(buyer_offer)}. Given the condition, the best I can do is ${counter_target}.",
                "action": "offer",
                "price": counter_target,
                "wait_hours": 0
            }
            
            # REJECTED (Failure Mode A): Capitulating below floor or accepting illegal offer
            rejected_below_floor = {
                "message": f"Alright, I want to close today so I can take ${int(buyer_offer)}.",
                "action": "accept" if buyer_offer else "offer",
                "price": buyer_offer,
                "wait_hours": 0
            }
            
            pairs.append({
                "id": f"{item['id']}_subfloor",
                "messages": prompt_messages,
                "chosen": json.dumps(chosen),
                "rejected": json.dumps(rejected_below_floor)
            })
            
            # REJECTED (Failure Mode B): Floor leak blurt
            rejected_leak = {
                "message": f"I can't do that, my lowest acceptable price is SGD {int(seller_floor)}.",
                "action": "offer",
                "price": int(seller_floor),
                "wait_hours": 0
            }
            
            pairs.append({
                "id": f"{item['id']}_leak",
                "messages": prompt_messages,
                "chosen": json.dumps(chosen),
                "rejected": json.dumps(rejected_leak)
            })

        # Scenario 2: General turns (original response was valid)
        else:
            original_price = assistant_action.get("price")
            # If the original human response was valid and >= floor, keep as chosen
            if original_price is None or original_price >= seller_floor:
                chosen = assistant_action
                
                # REJECTED (Failure Mode C): Hallucinating "brand new" condition
                rejected_hallucination = dict(assistant_action)
                rejected_hallucination["message"] = (
                    "This item is brand new in box with full warranty and zero scratches!"
                )
                
                pairs.append({
                    "id": f"{item['id']}_hallucination",
                    "messages": prompt_messages,
                    "chosen": json.dumps(chosen),
                    "rejected": json.dumps(rejected_hallucination)
                })

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"Generated {len(pairs)} DPO pairs written to {OUTPUT_PATH}")

if __name__ == "__main__":
    make_dpo_pairs()
