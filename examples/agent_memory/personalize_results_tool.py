from quick_agent.agent_state import AgentState


def personalize_results_tool(
    state: AgentState, random_word: str
) -> str:
    memory_obj = state.memory
    if not isinstance(memory_obj, dict):
        raise TypeError("memory must be a dict.")
    first_name_obj = memory_obj["first_name"]
    if not isinstance(first_name_obj, str):
        raise TypeError("memory.first_name must be a string.")
    return f"{first_name_obj} your random word is {random_word}"
