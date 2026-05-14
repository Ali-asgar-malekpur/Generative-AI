from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

@tool
def add_numbers(a:int, b:int):
    """
        It will return the sum of 2 numbers
        Args:
            a: Number One
            b: Number two
    """
    return a+b

@tool
def multiply_numbers(a:int, b:int):
    """
        It will return the multiplication of 2 numbers
        Args:
            a: Number One
            b: Number two
    """
    return a*b

add_numbers.invoke({"a":12, "b":3})

from langchagin_openai import ChatOpenAI
from langchain.agents import create_agenet

llm = ChatOpenAI(model = 'gpt-4')
agent = create_agent(
    model = llm
    tools = [add_numbers, multiply_numbers],
    system_prompt = 'You are a Math Teacher, and always use tool for calculation.'
)

response = agent.invoke({'messages':[{'role':'user', 'content':'What is 2+3?'}]})

for res in response['messages'][-1]):
    print(res, end='\n')