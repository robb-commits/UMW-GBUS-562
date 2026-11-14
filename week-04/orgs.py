"""Organization folders and their employee records, shared by agent.py and hr_mcp_server.py.

An organization folder holds four items:
    hr-docs/            the HR policy documents (HTML)
    employees.csv       one row per employee
    sample-emails/      example employee emails (.eml)
    system_prompt.txt   the agent's rules for this organization

employees.csv may use any column names. They are read from its header: the column with "email" in
its name holds the work address, the one with "name" the employee's name, and columns about days,
hours or balances (e.g. vacation_days_left, PTO Days Remaining) are leave balances. A blank or
non-numeric balance is reported as not on record, never as a number.
"""

import csv
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ORG = HERE / "hartwell-hr"
REQUIRED = ("hr-docs", "employees.csv", "sample-emails", "system_prompt.txt")
BALANCE_WORDS = {"day", "days", "hour", "hours", "balance", "remaining", "left"}


def resolve_org(value: str | Path | None) -> Path:
    """Return the organization folder, checking it has all four items. A bare folder name such as
    'ridgeline-hotels' is also looked for in week-04/. Raises ValueError with a readable message."""
    if value is None:
        return DEFAULT_ORG
    path = Path(value).expanduser()
    if not path.is_dir() and (HERE / path).is_dir():
        path = HERE / path
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"Organization folder not found: {value}")
    missing = [item for item in REQUIRED if not (path / item).exists()]
    if missing:
        raise ValueError(f"{path.name} is missing {', '.join(missing)}. An organization folder needs: {', '.join(REQUIRED)}.")
    return path


def words(header: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", header.lower()).split())


class Employees:
    """An organization's employees.csv, with its columns worked out from the header."""

    def __init__(self, org: Path):
        self.path = org / "employees.csv"
        with self.path.open(newline="", encoding="utf-8-sig") as f:
            self.header = [h.strip() for h in (next(csv.reader(f), None) or [])]
        squashed = {h: re.sub(r"[^a-z]", "", h.lower()) for h in self.header}
        self.email_col = next((h for h in self.header if "email" in squashed[h]), None)
        self.name_col = next((h for h in self.header if "name" in squashed[h] and h != self.email_col), None)
        if self.email_col is None:
            raise ValueError(f"{self.path} has no email column in its header.")
        self.balance_cols = [h for h in self.header
                             if h not in (self.email_col, self.name_col) and words(h) & BALANCE_WORDS]
        self.detail_cols = [h for h in self.header
                            if h not in (self.email_col, self.name_col) and h not in self.balance_cols]

    def rows(self):
        # Read the file on every call, so edits to employees.csv take effect immediately.
        with self.path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                yield {(k or "").strip(): (v or "").strip() for k, v in row.items()}

    def find(self, email: str) -> dict | None:
        email = email.strip().lower()
        return next((r for r in self.rows() if r.get(self.email_col, "").lower() == email), None)

    def names(self) -> dict[str, str]:
        return {r[self.email_col].lower(): r.get(self.name_col, "") for r in self.rows() if r.get(self.email_col)}

    def record(self, row: dict) -> dict:
        """One employee's record as the balance tool returns it."""
        balances, not_recorded = {}, []
        for col in self.balance_cols:
            number = to_number(row.get(col, ""))
            if number is None:
                not_recorded.append(col)
            else:
                balances[col] = number
        return {
            "email": row[self.email_col],
            "name": row.get(self.name_col, "") if self.name_col else None,
            "details": {col: row.get(col) or None for col in self.detail_cols},
            "leave_balances": balances,
            "not_on_record": not_recorded,
            "note": ("These are the only leave balances on record for this employee. Don't state, estimate "
                     "or guess a balance for any leave type that isn't in leave_balances."),
        }


def to_number(value: str) -> int | float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):  # "nan" or "inf" isn't a balance
        return None
    return int(number) if number.is_integer() else number
