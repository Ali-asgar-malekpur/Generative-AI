from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = 'gpt-4')
question = 'Can you explain me GenAI?'
#This takes time since entire ans is being prepared and once done, it shows altogether.

#Instead we can stream the data to show what is generated
# reponse = llm.invoke(question)

reponse = llm.stream(question) #response here is object/iterator
for chunk in response:
    print(chunk.content, end="")
print(respose.content)

#Using Groq---------------------

from langchain_groq import ChatGroq
llm = ChatGroq(model = 'ppenai/gpt-oss-20b', streaming = True)
question = 'can you explain me GEN-AI?'
response = llm.stream(question)
for chunk in response:
    print(chunk.content, end='')