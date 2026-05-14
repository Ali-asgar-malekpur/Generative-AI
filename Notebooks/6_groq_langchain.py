from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm = ChatGroq(model = 'openai/gpt-oss-120b')
response = llm.invoke('Explain me about deep learning in detail!')
print(response.content) #O/p will be received in 2-5 seconds cause it's fast. Instead if we use normal ChatGpt Models, they would take 40+ seconds.