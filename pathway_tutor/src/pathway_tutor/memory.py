from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory

class PathwayMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def save_context(self, input_str, output_str):
        self.memory.save_context(
            {"input": input_str},
            {"output": output_str}
        )
    
    def load_memory(self):
        return self.memory.load_memory_variables({})

    def get_history(self, num_exchanges=3):
        history = self.load_memory()["chat_history"]
        return "\n".join(
            [f"{msg.content}" for msg in history[-num_exchanges*2:]]
        )