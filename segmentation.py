import os
from pathlib import Path
from typing_extensions import Annotated

import autogen
from autogen import ConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

from jinja2 import Environment, FileSystemLoader

from dotenv import load_dotenv

load_dotenv()


config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

llm_config = {
    "config_list": config_list,
    "timeout": 60,
    "temperature": 0.8,
    "seed": 1234,
}

# Get the directory of the current script
script_dir = Path(__file__).parent

# Define the path to the 'prompts' directory
prompts_dir = script_dir / "prompts"

# Set up Jinja environment and template loader
env = Environment(loader=FileSystemLoader(prompts_dir))

analysis_generator_template = env.get_template("analysis_generator.j2")
analysis_generator_prompt = analysis_generator_template.render({})

analysis_reviewer_template = env.get_template("analysis_reviewer.j2")
analysis_reviewer_prompt = analysis_reviewer_template.render({})


lawer_aid = RetrieveUserProxyAgent(
    name="Lawer_assistant",
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    # max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "qa",
        "docs_path": "D:\\Personal\\data",
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "chroma",
        "overwrite": True,  # set to True if you want to overwrite an existing collection
        "get_or_create": True,  # set to False if don't want to reuse an existing collection
    },
    code_execution_config=False,  # set to False if you don't want to execute the code
    description="Assistant who has extra content retrieval power for solving difficult problems.",
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

PROBLEM = "List all the points presented by defense which led to acquittal"


def _reset_agents():
    lawer_aid.reset()
    analysis_generator.reset()
    analysis_reviewer.reset()


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the output folder path
output_folder = os.path.join(script_dir, "output")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)


def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[lawer_aid, analysis_generator, analysis_reviewer],
        messages=[],
        max_round=12,
        # speaker_selection_method="round_robin",
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    lawer_aid.initiate_chat(
        manager,
        message=lawer_aid.message_generator,
        problem=PROBLEM,
        n_results=3,
    )

    # print(lawer_aid.chat_messages)
    # analysis = None
    # review = None

    analysis_index = 0
    review_index = 0

    for msg_lst in lawer_aid.chat_messages.values():
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


rag_chat()
