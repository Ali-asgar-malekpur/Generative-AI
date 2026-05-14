prompt = [
    {'role':'user', 'content':'Hello My Name is Ali asgar'},
    {'role':'ai', 'content':'Hello, Ali asgar How can I assist you today?'},
    {'role':'user', 'content':'What is my Name?'},
    ] #O/p: Your name is Ali asgar

res = llm.invoke(prompt)
print(res.content)

history = []
while True:
    query = input('User: ')
    if query.lower() in ('exit', 'quit', 'bye'):
        print("Bye Bye")
        break

    history.append({'role':'user', 'content':query})
    print("user: ", query)
    
    res = llm.invoke(history)
    history.append({'role':'ai', 'content':res.content})
    
    print("AI: ", res.content, "\n")
    