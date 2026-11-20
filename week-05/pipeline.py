"""Coordinator 1: a fixed pipeline. Code, not Claude, decides the steps, always in this order:

  1. the supply analyst writes its brief
  2. for each order at risk, a customer notice (skipped if no order is at risk)
  3. a reply to the supplier

Every draft goes through the review step in specialists.py (reviewer, up to 2 revisions, then the
name check in code). --org picks the company folder in week-05 (default brookfield-bikes). Drafts are saved to
week-05/drafts/<org>/pipeline/<email>/. Nothing is sent. At the
end the program prints a summary: drafts saved, drafts marked for a person, decisions for a person,
and total tokens.

Usage (from the top of the repository):
    .venv/bin/python week-05/pipeline.py --email week-05/brookfield-bikes/supplier-emails/01-northgate-controller-delay.eml
    .venv/bin/python week-05/pipeline.py --email week-05/brookfield-bikes/supplier-emails/02-kendall-tire-options.eml \
        --save week-05/runs/pipeline-02.txt
    .venv/bin/python week-05/pipeline.py --org copperline-coffee \
        --email week-05/copperline-coffee/supplier-emails/01-danforth-pump-delay.eml
"""

import anthropic

from common import Email, run_cli
from specialists import Team, heading, print_start, print_summary


async def coordinate(client: anthropic.Anthropic, email: Email) -> None:
    print_start(email, "fixed pipeline")
    team = Team(client, email, "pipeline")

    heading("Step 1 of 3: supply analyst")
    brief = await team.run_analyst()

    heading("Step 2 of 3: customer notices")
    if brief["orders_at_risk"]:
        for order in brief["orders_at_risk"]:
            team.draft_customer_notice(order["order_id"])
    else:
        print("No orders are at risk, so this step is skipped.")

    heading("Step 3 of 3: supplier reply")
    team.draft_supplier_reply()

    print_summary(team)


if __name__ == "__main__":
    run_cli("Prepare a company's response to one supplier email with a fixed pipeline. Nothing is sent.", coordinate)
