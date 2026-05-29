from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain.schema import Document
import re
from loguru import logger
from typing import List, Dict
from functools import partial, lru_cache
from langgraph.graph import StateGraph, START, END, MessagesState

from ..prompts.security_assessment_assistant import (
    security_assessment_assistant_prompt_message,
)
from ..helpers.model_config import fetch_model_from_ollama
#from backend.fastapi.langgraph.helpers.vector_db_operations import (
#    setup_vectorstore_saa,
#    custom_numbered_header_split,
#    load_markdown,
#)
from backend.fastapi.langgraph.helpers.vector_db_operations import setup_vectorstore_saa

from langchain_core.tools.retriever import create_retriever_tool


def list_sections(split_docs: list[Document]) -> str:
    """Generate a formatted Markdown list of all available sections."""
    return "Available sections:\n" + "\n".join(
        f"- *Section {doc.metadata['section_number']}*: {doc.metadata['title']}"
        for doc in split_docs
        if "section_number" in doc.metadata
    )


def preprocess_query(query: str, section_map: dict) -> str:
    """Enhance queries that mention specific sections."""
    match = re.search(r"section\s*(\d+)", query, re.IGNORECASE)
    if match:
        section_num = match.group(1)
        if section_num in section_map:
            return f"{section_map[section_num]} {query}"
    return query


def is_section_listing_query(query: str) -> bool:
    """Check if the query is asking for a list of sections."""
    return "list" in query.lower() and "section" in query.lower()


def _initialize_security_assistant(
    file_name: str = "security_assessment_doc",
    input_file_path: str = "backend/fastapi/langgraph/input_files/SecurityAssessmentTemplate-Guide.md",
    embedding_model: str = "mxbai-embed-large",
    persist_dir: str = "backend/chromadb_vectorstore",
):
    try:
        vectorstore, split_docs = setup_vectorstore_saa(
            file_name=file_name,
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            input_file_path=input_file_path,
        )

        # 2. Create retriever tool
        retriever = create_retriever_tool(
            retriever=vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            name="security_assessment_retriever",
            description="Use this tool to retrieve information from the security assessment and explain each section.",
        )

        # 3. Create LLM with retriever tool
        llm = fetch_model_from_ollama("llama3.2", temperature=0.2)

        # 4. Get split_docs + section map using your existing splitting logic
        #split_docs = custom_numbered_header_split(load_markdown(input_file_path))
        section_map = {
            doc.metadata["section_number"]: doc.metadata["title"]
            for doc in split_docs
            if "section_number" in doc.metadata
        }

        return llm.bind_tools([retriever]), retriever, split_docs, section_map

    except Exception as e:
        logger.error(f"Error initializing security assistant context: {e}")
        raise


def security_assistant_node(
    state: MessagesState, llm, retriever, split_docs, section_map
) -> Dict[str, list]:
    """
    Security assistant node that processes a single message and returns the response.
    """
    try:
        messages = state["messages"]

        # Get the last human message
        last_message = messages[-1] if messages else None
        if not last_message or not isinstance(last_message, HumanMessage):
            logger.warning("No human message found in state")
            return {
                "messages": [
                    AIMessage(
                        content="I didn't receive a question. Please ask me something about the security assessment."
                    )
                ]
            }

        user_input = last_message.content

        # Handle "list sections" queries directly
        if is_section_listing_query(user_input):
            section_text = list_sections(split_docs)
            response = AIMessage(content=section_text)
            return {"messages": [response]}

        # Enhance query if it references "section X"
        processed_query = preprocess_query(user_input, section_map)

        # Update the last message with processed query if it was enhanced
        if processed_query != user_input:
            messages = messages[:-1] + [HumanMessage(content=processed_query)]

        # Invoke LLM with all messages
        response = llm.invoke(messages)
        responses_to_add = []

        if hasattr(response, "tool_calls") and response.tool_calls:
            # Add the initial response with tool calls
            responses_to_add.append(response)

            # Process each tool call
            for tool_call in response.tool_calls:
                tool_result = retriever.invoke(tool_call["args"])
                tool_msg = ToolMessage(
                    content=str(tool_result), tool_call_id=tool_call["id"]
                )
                responses_to_add.append(tool_msg)

            # Get final response after tool calls
            final_response = llm.invoke(messages + responses_to_add)
            responses_to_add.append(final_response)
        else:
            # No tool calls, just add the response
            responses_to_add.append(response)

        return {"messages": responses_to_add}

    except Exception as e:
        logger.error(f"Error in security_assistant_node: {e}")
        error_msg = AIMessage(
            content="I'm sorry, I'm having trouble accessing the security assessment information right now. Please try again."
        )
        return {"messages": [error_msg]}

@lru_cache(maxsize=1)
def create_security_assistant_graph():
    """Creates a compiled LangGraph for the security assistant chatbot."""
    try:
        llm, retriever, split_docs, section_map = _initialize_security_assistant()

        # Create node with context using partial
        node_with_context = partial(
            security_assistant_node,
            llm=llm,
            retriever=retriever,
            split_docs=split_docs,
            section_map=section_map,
        )

        builder = StateGraph(MessagesState)
        builder.add_node("security_assistant", node_with_context)
        builder.add_edge(START, "security_assistant")
        builder.add_edge("security_assistant", END)

        return builder.compile()

    except Exception as e:
        logger.error(f"Error creating security assistant graph: {e}")
        raise


def invoke_security_assistant_chat(
    messages: List[Dict[str, str]] = None, thread_id=None
) -> List[Dict[str, str]]:
    """
    Main function to invoke the security assistant chat graph.
    It manages the conversation state and returns the full history.
    """
    try:
        if messages is None:
            messages = []

        # Convert dict messages to LangChain message objects
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "human")
            content = msg.get("content", "")
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "human":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "ai":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        # Add system prompt if not present
        if not langchain_messages or not isinstance(
            langchain_messages[0], SystemMessage
        ):
            system_prompt = SystemMessage(
                content=security_assessment_assistant_prompt_message
            )
            langchain_messages.insert(0, system_prompt)

        # Create the graph
        graph = create_security_assistant_graph()

        # Invoke the graph with the current conversation history
        result = graph.invoke(
            {"messages": langchain_messages},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Get the final messages
        final_messages = result["messages"]

        # Convert LangChain message objects back to dictionaries for the response
        response_messages = []
        for msg in final_messages:
            role = "unknown"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "human"
            elif isinstance(msg, AIMessage):
                role = "ai"
            elif isinstance(msg, ToolMessage):
                # Skip tool messages in the response as they're internal
                continue

            response_messages.append({"role": role, "content": msg.content})

        return response_messages

    except Exception as e:
        logger.error(f"Error in invoke_security_assistant_chat: {e}")
        return [
            {
                "role": "ai",
                "content": "I apologize, but I encountered a critical error. Please try again.",
            }
        ]


if __name__ == "__main__":
    logger.info(
        "Not a runnable file. To run the business owner, please use api or test files"
    )
