from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
import re
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# 模型
llm = ChatOllama(
    model = "deepseek-r1:1.5b",
    temperature = 0,
    base_url="http://21.6.90.166.devcloud.woa.com:11434"
)



# 定义角色模板
system_template = "你是一个客服助手，回答关于 {product} 的问题。"
human_template = "用户询问：{question}"

# 创建模板实例
system_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 组合成聊天模板
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

print(chat_prompt.input_variables)

# 填充变量并生成提示
# prompt = chat_prompt.format(product="姓名", question="张三的名字是什么")
# print(prompt)


chain = chat_prompt | llm
res = chain.invoke({"product": "姓名","question":"张三的名字是什么"})
print(res.content)

# chunks = []
# for chunk in chain.stream({"input": "张三"}):
#     chunks.append(chunk)
#     # 打印每个块的内容
#     print(chunk, end="|", flush=True)