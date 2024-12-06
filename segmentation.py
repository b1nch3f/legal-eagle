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


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


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

legal_assistant_template = env.get_template("legal_assistant.j2")
legal_assistant_prompt = legal_assistant_template.render({})

verification_assistant_template = env.get_template("verification_assistant.j2")
verification_assistant_prompt = verification_assistant_template.render({})


lawer_aid = RetrieveUserProxyAgent(
    name="Lawer_assistant",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    max_consecutive_auto_reply=3,
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
    is_termination_msg=termination_msg,
    system_message=legal_assistant_prompt,
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
    human_input_mode="NEVER",  # never ask for human input
)

analysis_reviewer = ConversableAgent(
    name="analysis_reviewer",
    # is_termination_msg=termination_msg,
    system_message=verification_assistant_prompt,
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


def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[lawer_aid, analysis_generator, analysis_reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
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
    analysis = None
    review = None

    for val in lawer_aid.chat_messages.values():
        for item in val:
            content, name = item["content"], item["name"]
            if name == "analysis_generator":
                analysis = content
            elif name == "analysis_reviewer":
                review = content

    return analysis, review


analysis, review = rag_chat()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the output folder path
output_folder = os.path.join(script_dir, "output")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)


# Saving analysis
file_name = f"analysis.txt"
file_path = os.path.join(output_folder, file_name)

with open(file_path, "w") as output:
    output.write(analysis)

# Saving review
file_name = f"review.txt"
file_path = os.path.join(output_folder, file_name)

with open(file_path, "w") as output:
    output.write(review)
