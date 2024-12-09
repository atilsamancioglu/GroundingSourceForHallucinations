from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

model = ChatOpenAI(model="gpt-4")

store = {}


def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ground your response in factual data from your pre-training set, specifically referencing or quoting authoritative sources when possible."),
        MessagesPlaceholder(variable_name="messages")
    ]

)

chain = prompt | model
config = {"configurable" : {"session_id" : "acbde123"}}
with_message_history = RunnableWithMessageHistory(chain, get_session_history)


if __name__ == '__main__':
    while True:
        source = input("\n> Source: ")
        question = input("\n> Question: ")
        for r in with_message_history.stream(
            [
                HumanMessage(content=f"Respond to this question using only information that can be attributed to {source}. {question}")
                ],
                config = config,
        ):
            print(r.content, end=" ")