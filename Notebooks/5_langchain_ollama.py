from langchain_ollama import ChatOllama
llm = ChatOllama(model = 'gemma3:1b') #Only use the Model that is installed in your PC
response = llm.invoke('Hello, I am Ali asgar')
print(response.content)