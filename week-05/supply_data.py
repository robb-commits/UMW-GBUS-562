"""Brookfield Bicycle Company's supply data, and the coverage calculation the supply analyst relies on.

The data is the five CSV files in week-05/brookfield-bikes, read fresh on every call. Days are counted
from today (day 0): stock on hand is available on day 0, and a shipment can be used from the day it
arrives.

Coverage rule: orders are filled in order of ship day, earliest first (ties by order number), across
every bike model that uses the part. An order ships only when all of its parts are available, so an
order that is partly covered is still late. This file makes no Claude calls.
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "brookfield-bikes"


class DataError(ValueError):
    """A request the data can't answer, such as an unknown part."""


def read(name: str) -> list[dict]:
    with (DATA / name).open(newline="", encoding="utf-8-sig") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def find_part(part_id: str) -> dict:
    part_id = part_id.strip().upper()
    part = next((p for p in read("parts.csv") if p["part_id"].upper() == part_id), None)
    if part is None:
        known = ", ".join(p["part_id"] for p in read("parts.csv"))
        raise DataError(f"No part {part_id!r}. Known parts: {known}.")
    return part


def scheduled_arrivals(part_id: str) -> list[dict]:
    return [{"po_number": s["po_number"], "supplier": s["supplier"], "qty": int(s["qty"]),
             "arrives_in_days": int(s["arrives_in_days"])}
            for s in read("inbound.csv") if s["part_id"] == part_id]


def get_part(part_id: str) -> dict:
    part = find_part(part_id)
    pid = part["part_id"]
    return {
        "part_id": pid,
        "description": part["description"],
        "supplier": part["supplier"],
        "on_hand": int(part["on_hand"]),
        "scheduled_shipments": scheduled_arrivals(pid),
        "used_in": [{"model": b["model"], "qty_per_bike": int(b["qty_per_bike"])}
                    for b in read("bill-of-materials.csv") if b["part_id"] == pid],
    }


def get_orders_for_part(part_id: str) -> dict:
    """Open orders whose bike model uses the part, earliest ship day first. Contact details are
    left out: the analyst doesn't need them."""
    pid = find_part(part_id)["part_id"]
    per_bike = {b["model"]: int(b["qty_per_bike"]) for b in read("bill-of-materials.csv") if b["part_id"] == pid}
    orders = [{"order_id": o["order_id"], "customer": o["customer"], "model": o["model"],
               "bikes": int(o["qty"]), "units_needed": int(o["qty"]) * per_bike[o["model"]],
               "ship_day": int(o["ship_in_days"]), "late_penalty": o["late_penalty"]}
              for o in read("orders.csv") if o["model"] in per_bike]
    orders.sort(key=lambda o: (o["ship_day"], o["order_id"]))
    return {"part_id": pid, "orders": orders}


def check_coverage(part_id: str, arrivals: list[dict] | None = None) -> dict:
    """Which orders can ship on time, given the part's stock on hand and its arrivals.

    arrivals is the complete schedule of shipments still to come for this part, as
    [{"qty": ..., "arrives_in_days": ...}], replacing the scheduled shipments in inbound.csv.
    If it is None, the scheduled shipments are used.
    """
    part = get_part(part_id)
    if arrivals is None:
        schedule = [{"qty": s["qty"], "arrives_in_days": s["arrives_in_days"]} for s in part["scheduled_shipments"]]
        source = "scheduled shipments in inbound.csv"
    else:
        schedule = []
        for a in arrivals:
            qty, day = a.get("qty"), a.get("arrives_in_days")
            if not (isinstance(qty, int) and isinstance(day, int)) or isinstance(qty, bool) or isinstance(day, bool):
                raise DataError("Each arrival needs whole numbers: qty and arrives_in_days.")
            if qty <= 0 or day < 0:
                raise DataError("Each arrival needs qty above 0 and arrives_in_days of 0 or more (days from today).")
            schedule.append({"qty": qty, "arrives_in_days": day})
        source = "arrivals given to this tool"
    schedule.sort(key=lambda a: a["arrives_in_days"])

    def supply_by(day: int) -> int:
        return part["on_hand"] + sum(a["qty"] for a in schedule if a["arrives_in_days"] <= day)

    days = [0] + [a["arrives_in_days"] for a in schedule]
    total_supply = supply_by(max(days))
    needed_so_far = 0
    results = []
    for order in get_orders_for_part(part["part_id"])["orders"]:
        needed_so_far += order["units_needed"]
        available = supply_by(order["ship_day"]) - (needed_so_far - order["units_needed"])
        units_short = min(order["units_needed"], max(0, order["units_needed"] - available))
        ready_day = next((d for d in sorted(set(days)) if supply_by(d) >= needed_so_far), None)
        results.append({
            "order_id": order["order_id"], "customer": order["customer"], "model": order["model"],
            "units_needed": order["units_needed"], "ship_day": order["ship_day"],
            "covered": units_short == 0, "ready_day": ready_day,
            "days_late": None if ready_day is None else max(0, ready_day - order["ship_day"]),
            "units_short": units_short, "late_penalty": order["late_penalty"],
        })
    return {
        "part_id": part["part_id"],
        "on_hand": part["on_hand"],
        "arrivals_used": schedule,
        "arrivals_source": source,
        "total_supply": total_supply,
        "total_needed": needed_so_far,
        "orders": results,
        "orders_at_risk": [r["order_id"] for r in results if not r["covered"]],
        "note": ("Orders are filled complete, earliest ship day first, so a short order holds up the ones after it. "
                 "covered is true if the order can ship on its ship day. units_short is how many of an order's parts "
                 "won't be available by its ship day; ready_day is the first day all of them are "
                 "(null if the arrivals never cover it). Days are counted from today."),
    }


def find_order(order_id: str) -> dict | None:
    """One order's row from orders.csv, including the customer's contact email."""
    return next((o for o in read("orders.csv") if o["order_id"].upper() == order_id.strip().upper()), None)
