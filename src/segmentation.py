import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from typing_extensions import Annotated

import autogen
from autogen import Agent
from autogen import ConversableAgent
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from search import single_vector_search

service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
key = os.environ["AZURE_SEARCH_API_KEY"]
k_nearest_neighbors = 20

from jinja2 import Environment, FileSystemLoader

from dotenv import load_dotenv

load_dotenv()

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_type": "azure",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": "2024-08-01-preview",
    }
]

llm_config = {
    "config_list": config_list,
    "timeout": 60,
    "temperature": 0.8,
    "seed": 1234,
}

prompts_dir = os.path.join("prompts")

# Set up Jinja environment and template loader
env = Environment(loader=FileSystemLoader(prompts_dir))

analysis_generator_template = env.get_template("analysis_generator.j2")
analysis_generator_prompt = analysis_generator_template.render({})

analysis_reviewer_template = env.get_template("analysis_reviewer.j2")
analysis_reviewer_prompt = analysis_reviewer_template.render({})

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the output folder path
output_folder = os.path.join(os.path.dirname(script_dir), "output")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

PROMPT_QA = """Answer user's questions based on the
context provided by the user.

User's question is: {input_question}

Context is: {input_context}
"""


class RetrieverAgent(ConversableAgent):
    @staticmethod
    def message_generator(sender, recipient, context):
        problem = context.get("problem", "")

        chunks = single_vector_search(problem)

        result = ""

        for i, item in enumerate(chunks, 1):
            result += f"{i}. {item['chunk']}\n"

        result = PROMPT_QA.format(input_question=problem, input_context=result)

        return result


retriever_agent = RetrieverAgent(
    "retriever_agent",
    llm_config=False,  # no LLM used for human proxy
    human_input_mode="NEVER",  # always ask for human input
)

analysis_generator = ConversableAgent(
    name="analysis_generator",
    system_message=analysis_generator_prompt,
    # max_consecutive_auto_reply=10,
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
    human_input_mode="NEVER",  # never ask for human input
)

analysis_reviewer = ConversableAgent(
    name="analysis_reviewer",
    system_message=analysis_reviewer_prompt,
    # max_consecutive_auto_reply=10,
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
    human_input_mode="NEVER",  # never ask for human input
)


def _reset_agents():
    retriever_agent.reset()
    analysis_generator.reset()
    analysis_reviewer.reset()


if __name__ == "__main__":
    _reset_agents()

    PROBLEM = "List all the winning argumnets presented by defense"

    groupchat = autogen.GroupChat(
        agents=[retriever_agent, analysis_generator, analysis_reviewer],
        messages=[],
        max_round=2,
        speaker_selection_method="round_robin",
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # initial chat
    retriever_agent.initiate_chat(
        manager,
        message=retriever_agent.message_generator,
        problem=PROBLEM,
    )

    analysis_index = 0
    review_index = 0

    for msg_lst in retriever_agent.chat_messages.values():
        for msg in msg_lst:
            content, name = msg["content"], msg["name"]
            if name == "analysis_generator":
                # Saving analysis
                file_name = f"analysis_{analysis_index}.txt"
                file_path = os.path.join(output_folder, file_name)

                with open(file_path, "w") as output:
                    output.write(content)

                analysis_index += 1
            elif name == "analysis_reviewer":
                # Saving review
                file_name = f"review_{review_index}.txt"
                file_path = os.path.join(output_folder, file_name)

                with open(file_path, "w") as output:
                    output.write(content)

                review_index += 1
