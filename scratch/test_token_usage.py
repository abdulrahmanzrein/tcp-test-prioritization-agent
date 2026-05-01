
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

def test_usage():
    model = init_chat_model("gpt-4o-mini", temperature=0)
    response = model.invoke([HumanMessage(content="Hello")])
    print(f"Response type: {type(response)}")
    print(f"Usage metadata: {getattr(response, 'usage_metadata', 'N/A')}")
    if hasattr(response, 'additional_kwargs'):
        print(f"Additional kwargs: {response.additional_kwargs.get('token_usage', 'N/A')}")

if __name__ == "__main__":
    test_usage()
