"""An MCP server with Brookfield Bicycle Company's three supply tools, for the supply analyst.

  get_part             a part's stock on hand, scheduled shipments, and the bike models that use it
  get_orders_for_part  the open customer orders that need the part, earliest ship day first
  check_coverage       which of those orders can ship on time with a given arrival schedule

The tools only read week-05/brookfield-bikes; the calculations are in supply_data.py. The server
talks MCP over stdin/stdout, so an MCP client starts it. It makes no Claude calls and needs no key.

Usage (from the top of the repository; normally specialists.py starts it):
    .venv/bin/python week-05/supply_mcp_server.py
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

import supply_data

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


class Arrival(BaseModel):
    qty: int = Field(description="How many units arrive in this shipment.")
    arrives_in_days: int = Field(description="When it arrives, in days from today (0 = today).")


def create_server() -> MCPServer:
    server = MCPServer(
        "brookfield-supply",
        instructions="Brookfield Bicycle Company's parts, inbound shipments and customer orders, read-only.",
        log_level="WARNING",
    )

    @server.tool(annotations=READ_ONLY)
    def get_part(part_id: str) -> dict:
        """Get one part's description, supplier, stock on hand, scheduled shipments (purchase order,
        quantity, arrival in days from today), and the bike models that use it with how many each
        bike needs."""
        try:
            return supply_data.get_part(part_id)
        except supply_data.DataError as e:
            raise ToolError(str(e))

    @server.tool(annotations=READ_ONLY)
    def get_orders_for_part(part_id: str) -> dict:
        """List the open customer orders whose bike model uses this part, earliest ship day first:
        order number, customer, model, bikes, units of the part needed, ship day (days from today)
        and whether the order has a late penalty."""
        try:
            return supply_data.get_orders_for_part(part_id)
        except supply_data.DataError as e:
            raise ToolError(str(e))

    @server.tool(annotations=READ_ONLY)
    def check_coverage(part_id: str, arrivals: list[Arrival] | None = None) -> dict:
        """Work out which orders for a part can ship on time. Orders are filled earliest ship day
        first from stock on hand plus arrivals. An order ships only when all its units are there, so a
        short order holds up the ones after it.

        arrivals: the complete schedule of shipments still to come for this part, in days from today.
        It replaces the scheduled shipments, so include every shipment, not only the changed one.
        Leave it out to check the current schedule. To compare a supplier's options, call this once
        per option.

        Returns, for each order: whether it is covered (can ship on its ship day), and if not, the
        units short and ready_day, the first day it can ship; also units needed, ship day, days late
        and late penalty."""
        try:
            given = None if arrivals is None else [a.model_dump() for a in arrivals]
            return supply_data.check_coverage(part_id, given)
        except supply_data.DataError as e:
            raise ToolError(str(e))

    return server


if __name__ == "__main__":
    create_server().run("stdio")
