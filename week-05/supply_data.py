"""A company's supply data, and the coverage calculation the supply analyst relies on.

The data is the five CSV files in one company folder under week-05 (brookfield-bikes by default; choose
another with set_org, e.g. copperline-coffee), read fresh on every call. Days are counted from today
(day 0): stock on hand is available on day 0, and a shipment can be used from the day it arrives.

Column names differ between companies (Brookfield's orders have model, qty and ship_in_days; Copperline's
have product, units and ship_by_day), so each file's columns are found from its header row: see COLUMNS.

Coverage rule: the arrivals given replace the part's scheduled shipments. Orders are filled complete, in
order of ship day, earliest first (ties by order number), across every product that uses the part. An
order ships only when all of its parts are available, so a short order holds up the ones after it. This
file makes no Claude calls.
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ORG = "brookfield-bikes"
DATA = HERE / DEFAULT_ORG
REQUIRED_FILES = ("parts.csv", "bill-of-materials.csv", "inbound.csv", "orders.csv", "suppliers.csv")


class DataError(ValueError):
    """A request the data can't answer, such as an unknown part, or a file whose columns can't be found."""


# ----- choosing the company ------------------------------------------------------------------

def org_names() -> list[str]:
    """Company folders in week-05: each has the five CSV files and a prompts folder."""
    return sorted(p.name for p in HERE.iterdir()
                  if p.is_dir() and (p / "prompts").is_dir() and all((p / f).is_file() for f in REQUIRED_FILES))


def set_org(org: str) -> None:
    global DATA
    if org not in org_names():
        raise DataError(f"No company folder {org!r} in week-05. Available: {', '.join(org_names()) or 'none'}.")
    DATA = HERE / org


def org() -> str:
    return DATA.name


def company() -> str:
    """The company's short name, from its folder: brookfield-bikes -> Brookfield."""
    return DATA.name.split("-")[0].capitalize()


# ----- reading the files, with columns found from their headers ------------------------------

# For each file, the logical fields the code uses: the header names known to mean each one, and a test
# that recognizes a new spelling. Fields not listed here (part_id, customer, ...) are read by their own name.
COLUMNS = {
    "bill-of-materials.csv": {
        "item": (("model", "product", "item", "sku"), lambda h: "model" in h or "product" in h),
        "per_item": (("qty_per_bike", "per_unit", "qty_per_unit", "quantity_per_unit"), lambda h: "per" in h),
    },
    "orders.csv": {
        "item": (("model", "product", "item", "sku"), lambda h: "model" in h or "product" in h),
        "quantity": (("qty", "units", "quantity"), lambda h: "qty" in h or "quantity" in h or h.startswith("unit")),
        "ship_day": (("ship_in_days", "ship_by_day", "ship_day"), lambda h: "ship" in h and "day" in h),
        "late_penalty": (("late_penalty",), lambda h: "penalty" in h),
    },
}


def read(name: str) -> list[dict]:
    with (DATA / name).open(newline="", encoding="utf-8-sig") as f:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def header(name: str) -> list[str]:
    with (DATA / name).open(newline="", encoding="utf-8-sig") as f:
        return [h.strip() for h in next(csv.reader(f), [])]


def columns(name: str) -> dict[str, str]:
    """The header name of each logical field in COLUMNS[name], for the current company."""
    found, heads = {}, header(name)
    for field, (known, looks_like) in COLUMNS.get(name, {}).items():
        match = next((h for h in heads if h.lower() in known), None)
        if match is None:
            guesses = [h for h in heads if looks_like(h.lower()) and h not in found.values()]
            match = guesses[0] if len(guesses) == 1 else None
        if match is None:
            raise DataError(f"{DATA.name}/{name}: can't tell which column is the {field.replace('_', ' ')}. "
                            f"Columns: {', '.join(heads)}.")
        found[field] = match
    return found


def item_label() -> str:
    """What the company calls the thing it sells, from the orders header: model or product."""
    return columns("orders.csv")["item"]


def orders() -> list[dict]:
    """Every order, with the logical fields: order_id, customer, contact_email, item, quantity,
    ship_day, late_penalty."""
    col = columns("orders.csv")
    return [{"order_id": o["order_id"], "customer": o["customer"], "contact_email": o.get("contact_email", ""),
             "item": o[col["item"]], "quantity": int(o[col["quantity"]]), "ship_day": int(o[col["ship_day"]]),
             "late_penalty": o[col["late_penalty"]]}
            for o in read("orders.csv")]


def bill_of_materials(part_id: str) -> list[dict]:
    col = columns("bill-of-materials.csv")
    return [{"item": b[col["item"]], "per_item": int(b[col["per_item"]])}
            for b in read("bill-of-materials.csv") if b["part_id"] == part_id]


# ----- the three tools' calculations ---------------------------------------------------------

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
    col = columns("bill-of-materials.csv")  # report the company's own column names
    return {
        "part_id": pid,
        "description": part["description"],
        "supplier": part["supplier"],
        "on_hand": int(part["on_hand"]),
        "scheduled_shipments": scheduled_arrivals(pid),
        "used_in": [{col["item"]: b["item"], col["per_item"]: b["per_item"]} for b in bill_of_materials(pid)],
    }


def get_orders_for_part(part_id: str) -> dict:
    """Open orders whose product uses the part, earliest ship day first. Contact details are left
    out: the analyst doesn't need them."""
    pid = find_part(part_id)["part_id"]
    per_item = {b["item"]: b["per_item"] for b in bill_of_materials(pid)}
    label = item_label()
    result = [{"order_id": o["order_id"], "customer": o["customer"], label: o["item"],
               "quantity": o["quantity"], "units_needed": o["quantity"] * per_item[o["item"]],
               "ship_day": o["ship_day"], "late_penalty": o["late_penalty"]}
              for o in orders() if o["item"] in per_item]
    result.sort(key=lambda o: (o["ship_day"], o["order_id"]))
    return {"part_id": pid, "orders": result}


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
    label = item_label()
    needed_so_far = 0
    results = []
    for order in get_orders_for_part(part["part_id"])["orders"]:
        needed_so_far += order["units_needed"]
        available = supply_by(order["ship_day"]) - (needed_so_far - order["units_needed"])
        units_short = min(order["units_needed"], max(0, order["units_needed"] - available))
        ready_day = next((d for d in sorted(set(days)) if supply_by(d) >= needed_so_far), None)
        results.append({
            "order_id": order["order_id"], "customer": order["customer"], label: order[label],
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
    """One order with its logical fields, including the customer's contact email."""
    return next((o for o in orders() if o["order_id"].upper() == order_id.strip().upper()), None)
