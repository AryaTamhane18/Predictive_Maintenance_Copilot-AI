from mcp.server.fastmcp import FastMCP
from tools.anomaly import detect_anomaly
from tools.rag import search_manuals
from tools.history import get_history_summary, init_db

# Create MCP server
mcp = FastMCP("Predictive Maintenance Tools")

# Initialise database on startup
init_db()


@mcp.tool()
def detect_anomaly_tool(machine_id: str = "bearing_1",
                        file_index: int = -1) -> dict:
    """
    Detect anomalies in bearing sensor data using Z-score analysis.
    Returns whether an anomaly exists, which channel is affected,
    and the Z-score value.
    """
    result = detect_anomaly(machine_id=machine_id, file_index=file_index)
    return {
        "machine_id": result.machine_id,
        "is_anomalous": result.is_anomalous,
        "max_z_score": result.max_z_score,
        "channel": result.channel,
        "severity": result.severity,
        "anomaly_count": result.anomaly_count,
        "total_readings": result.total_readings,
        "file_index": result.file_index,
        "total_files": result.total_files,
        "message": result.message
    }


@mcp.tool()
def search_manuals_tool(query: str) -> dict:
    """
    Search the SKF bearing maintenance manual for relevant sections.
    Use this to find recommended actions for detected faults.
    """
    result = search_manuals(query=query)
    return {
        "relevant_section": result.relevant_section,
        "page_number": result.page_number,
        "confidence": result.confidence
    }


@mcp.tool()
def get_history_tool(machine_id: str) -> str:
    """
    Get past maintenance history for a machine.
    Use this to check if similar faults have occurred before.
    """
    return get_history_summary(machine_id=machine_id)


if __name__ == "__main__":
    print("Starting MCP tool server...")
    print("Available tools:")
    print("  - detect_anomaly_tool")
    print("  - search_manuals_tool")
    print("  - get_history_tool")
    mcp.run()