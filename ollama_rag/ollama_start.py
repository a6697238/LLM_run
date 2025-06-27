from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

llm = Ollama(model="qwen2.5",base_url="http://9.135.120.238:11434")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能机器人"),
    ("user", "{input}")
])
chain = prompt | llm | output_parser

input_str = '''
    侯恬恬昨天早上吃的小米粥300g，鸡蛋1个;中午吃的豆芽200g，红烧肉100g;晚上吃的包子3个
    请问侯恬恬昨天的营养含量多少
'''

print(chain.invoke({"input": input_str}))
