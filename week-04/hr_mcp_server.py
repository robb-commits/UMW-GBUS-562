"""An MCP server that gives AI apps two Hartwell Foods HR tools.

  search_policy      returns the 4 HR document chunks closest to a search query, with source files
  get_leave_balance  returns one employee's vacation, sick and personal days from employees.csv

get_leave_balance has two modes:
  - HR staff mode (default): it takes an employee's work email.
  - Employee mode (--as-employee ADDRESS): it takes no inputs and can return only that one
    employee's balance, so the model can't look anyone else up.

Usage (from the top of the repository):
    .venv/bin/python week-04/hr_mcp_server.py                                   # HR staff mode
    .venv/bin/python week-04/hr_mcp_server.py --as-employee dana.whitfield@hartwellfoods.example
    .venv/bin/python week-04/hr_mcp_server.py --rebuild                         # rebuild the database, then exit

The server talks MCP over stdin/stdout, so an MCP client such as Claude Code starts it; it isn't
meant to be used by typing into it. It makes no Claude API calls and needs no API key.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import hr_search

EMPLOYEES_CSV = hr_search.HERE / "hartwell-hr" / "employees.csv"
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
NO_RECORD = ("No employee record matches {who}. Do not state, estimate or guess a leave balance. "
             "Say that HR will check the employee's record and follow up.")


def normalize_email(value: str) -> str:
    """Accept a bare address or a From line such as 'Dana Whitfield <dana@...>'."""
    match = re.search(r"<([^>]+)>", value)
    return (match.group(1) if match else value).strip().lower()


def find_employee(email: str) -> dict | None:
    # Read the file on every call, so edits to employees.csv take effect immediately.
    with EMPLOYEES_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["email"].strip().lower() == email:
                return row
    return None


def balance(email: str) -> dict:
    row = find_employee(normalize_email(email))
    if row is None:
        raise ToolError(NO_RECORD.format(who=repr(email)))
    return {
        "email": row["email"],
        "name": row["name"],
        "department": row["department"],
        "manager": row["manager"] or None,
        "vacation_days_left": int(row["vacation_days_left"]),
        "sick_days_left": int(row["sick_days_left"]),
        "personal_days_left": int(row["personal_days_left"]),
    }


class EmployeeModeServer(MCPServer):
    """In employee mode, refuse a get_leave_balance call that passes any input (such as another
    person's email), instead of silently returning this employee's balance as if it answered it."""

    async def call_tool(self, name, arguments, context=None):
        if name == "get_leave_balance" and arguments:
            raise ToolError("get_leave_balance takes no inputs here and returns only the balance of the "
                            "employee this session is for. It can't look up any other employee.")
        return await super().call_tool(name, arguments, context)


def create_server(as_employee: str | None) -> MCPServer:
    server = (EmployeeModeServer if as_employee else MCPServer)(
        "hartwell-hr",
        instructions="Hartwell Foods HR tools: search the HR policy documents and look up leave balances.",
        log_level="WARNING",  # tool errors already reach the client; don't also log them to the terminal
    )

    @server.tool(annotations=READ_ONLY)
    def search_policy(query: str) -> dict:
        """Search Hartwell Foods' HR policy documents (leave, parental and family leave, remote work,
        benefits enrollment, reporting workplace concerns). Returns the 4 closest sections, each with
        its rank, similarity score (0 to 1), source file, section heading and full text. Cite the
        source_file for every policy fact you use. Results marked weak_match probably don't answer
        the question; if none of the results do, say the policies don't cover it."""
        if not query.strip():
            raise ToolError("The search query is empty. Pass a short description of the policy question.")
        results = hr_search.search(hr_search.get_collection(), query)
        return {"query": query, "results": results}

    if as_employee:
        def get_leave_balance() -> dict:
            return balance(as_employee)

        server.add_tool(
            get_leave_balance,
            annotations=READ_ONLY,
            description=("Get the leave balance of the employee this session is for: days left of "
                         "vacation, sick and personal leave, plus their department and manager. Takes no "
                         "inputs and can't look up any other employee."),
        )
    else:
        @server.tool(annotations=READ_ONLY)
        def get_leave_balance(email: str) -> dict:
            """For HR staff: get one employee's leave balance by their work email address (for
            example dana.whitfield@hartwellfoods.example). Returns days left of vacation, sick and
            personal leave, plus their department and manager. If no record matches, don't state a
            balance."""
            return balance(email)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Hartwell Foods HR MCP server (stdio).")
    parser.add_argument("--as-employee", metavar="ADDRESS",
                        help="employee mode: get_leave_balance takes no inputs and returns only this employee's balance")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the HR document database, then exit")
    args = parser.parse_args()

    if args.rebuild:
        hr_search.build()
        return

    as_employee = None
    if args.as_employee:
        as_employee = normalize_email(args.as_employee)
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", as_employee):
            sys.exit(f"--as-employee needs an email address, got {args.as_employee!r}")
        if find_employee(as_employee) is None:
            hr_search.log(f"Warning: no employee record matches {as_employee}; get_leave_balance will return an error.")

    hr_search.get_collection()  # build the database now if it's missing or out of date, before clients connect
    create_server(as_employee).run("stdio")


if __name__ == "__main__":
    main()
