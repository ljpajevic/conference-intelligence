import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from core.registry import (
    list_conferences,
    get_conference,
    add_conference,
    update_discovered_topics,
    update_cfp_url,
)


# tools


@tool
def tool_list_conferences(filter_type: str = "all") -> str:
    """
    List all conferences in the registry.
    filter_type: 'all' to list everything, or a venue type like 'networking', 'mobile', 'systems'
    """
    confs = list_conferences()
    if filter_type != "all":
        confs = [c for c in confs if c["venue_type"] == filter_type]
    if not confs:
        return "No conferences found."
    lines = []
    for c in confs:
        seed = ", ".join(c["topic_areas_seed"])
        discovered = ", ".join(c["topic_areas_discovered"])
        topics = discovered if discovered else seed
        lines.append(f"- {c['full_name']} ({c['name']}) | {c['venue_type']} | topics: {topics}")
    return "\n".join(lines)


@tool
def tool_get_conference(name: str) -> str:
    """
    Get full details for a specific conference by short name.
    Example: name='sigcomm' or name='mobicom'
    """
    conf = get_conference(name.lower())
    if not conf:
        return f"Conference '{name}' not found in registry."
    return (
        f"Name: {conf['full_name']} ({conf['name']})\n"
        f"Type: {conf['venue_type']}\n"
        f"URL: {conf['url']}\n"
        f"CFP URL: {conf['cfp_url']}\n"
        f"DBLP key: {conf['dblp_key']}\n"
        f"Seed topics: {', '.join(conf['topic_areas_seed'])}\n"
        f"Discovered topics: {', '.join(conf['topic_areas_discovered'])}\n"
        f"Typical deadline months: {conf['typical_months']}\n"
        f"Last scraped: {conf['last_scraped']}\n"
        f"Added by: {conf['added_by']}"
    )


@tool
def tool_add_conference(
    name: str,
    full_name: str,
    dblp_key: str,
    venue_type: str,
    url: str = None,
    cfp_url: str = None,
    topic_areas: str = "",
    typical_months: str = "",
) -> str:
    """
    Add a new conference to the registry.
    topic_areas: comma-separated string e.g. 'networking, protocols, SDN'
    typical_months: comma-separated integers e.g. '1, 6'
    """
    topics = [t.strip() for t in topic_areas.split(",") if t.strip()]
    months = []
    for m in typical_months.split(","):
        try:
            months.append(int(m.strip()))
        except ValueError:
            pass

    conf = add_conference(
        name=name,
        full_name=full_name,
        dblp_key=dblp_key,
        venue_type=venue_type,
        url=url,
        cfp_url=cfp_url,
        topic_areas_seed=topics,
        typical_months=months,
    )
    return f"Added '{conf['full_name']}' to registry successfully."


@tool
def tool_update_cfp(name: str, cfp_url: str) -> str:
    """
    Update the CFP URL for a conference.
    Example: name='sigcomm', cfp_url='https://sigcomm.org/2025/cfp'
    """
    conf = update_cfp_url(name.lower(), cfp_url)
    if not conf:
        return f"Conference '{name}' not found."
    return f"Updated CFP URL for {conf['full_name']}: {cfp_url}"


# agent

def build_registry_agent():
    llm = ChatOllama(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    tools = [
        tool_list_conferences,
        tool_get_conference,
        tool_add_conference,
        tool_update_cfp,
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are the Conference Registry Agent. "
            "You MUST use tools for EVERY question. "
            "NEVER answer from your own knowledge. "
            "If asked to list conferences, call tool_list_conferences. "
            "If asked about a specific conference, call tool_get_conference. "
            "If you do not call a tool, your answer is wrong."
        ),
    )

    return agent


# Conversational interface

def run_registry_agent():
    agent = build_registry_agent()
    print("\nConference Registry Agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        # extract final assistant message
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                if "<|python_tag|>" not in msg.content:  # skip raw tool calls
                    print(f"\nAgent: {msg.content}\n")
                    break


if __name__ == "__main__":
    run_registry_agent()
