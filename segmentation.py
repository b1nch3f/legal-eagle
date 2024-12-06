import os
from pathlib import Path

from autogen import ConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

from jinja2 import Environment, FileSystemLoader

from dotenv import load_dotenv

load_dotenv()

config_list = [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

# Get the directory of the current script
script_dir = Path(__file__).parent

# Define the path to the 'prompts' directory
prompts_dir = script_dir / "prompts"

# Set up Jinja environment and template loader
env = Environment(loader=FileSystemLoader(prompts_dir))

legal_assistant_template = env.get_template("legal_assistant.j2")
legal_assistant_prompt = legal_assistant_template.render({})

legal_query_generator_template = env.get_template("legal_query_generator.j2")
legal_query_generator_prompt = legal_query_generator_template.render({})

legal_assistant = ConversableAgent(
    name="assistant",
    system_message=legal_assistant_prompt,
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

legal_query_generator = ConversableAgent(
    name="assistant",
    system_message=legal_query_generator_prompt,
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)
legal_professional_proxy_agent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
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
)

# reset the assistants
legal_query_generator.reset()
legal_assistant.reset()

queries = legal_query_generator.generate_reply(
    messages=[{"content": "I want to extract winning arguments", "role": "user"}]
)
# print(queries.split("\n"))


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the output folder path
output_folder = os.path.join(script_dir, "output")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Saving queries
file_name = f"queries.txt"
file_path = os.path.join(output_folder, file_name)

with open(file_path, "w") as output:
    output.write(queries)

for index, qa_problem in enumerate(queries.split("\n")):
    # qa_problem = "List all the points presented by defense which led to aquittal"
    chat_result = legal_professional_proxy_agent.initiate_chat(
        legal_assistant,
        message=legal_professional_proxy_agent.message_generator,
        problem=qa_problem,
    )

    # print(chat_result.summary)

    # Dynamically save the memo_{index}.txt file
    file_name = f"memo_{index}.md"
    file_path = os.path.join(output_folder, file_name)

    with open(file_path, "w") as output:
        output.write(chat_result.summary)
