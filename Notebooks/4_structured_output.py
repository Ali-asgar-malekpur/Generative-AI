from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = 'gpt-4o')

text = 'Hello, My Name is Ali asgar Malekpur. Email is mralimalek77651@gmail.com and age is 25. Number is 7096790053'
response = llm.invoke(f'Pleaes give me only name, mail and age from this {text}')
print(type(response.content)) #Str, However i want structured Output mayve in JSON Format

from pydantic import BaseModel, Field
class ResponseStructure(BaseModel):
    name:str = Field(description = 'Name of the User')
    email:str = Field(description = 'Email Address')
    age:int = Field(description = 'Age')
    
class ResponseInList(BaseModel):
    all_in_list_example = List[ResponseStructure]
#structured_llm = llm.with_structured_output(ResponseInList)
    
structured_llm = llm.with_structured_output(ResponseStructure) #Output will be in llm again

response = structured_llm.invoke(f'Please give me only name and email')
print(response) #O/P: ResponseStructure(name='Ali asgar', email='mralimalek77651@gmail.com')
print(response.name) #O/P: Ali asgar
print(response.email) #O/P: mralimalek77651@gmail.com
print(response.model_dump()) #O/P: {'name':'Ali asgar', ...etc} Converting to key-value pair