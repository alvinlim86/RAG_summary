import os
from dotenv import load_dotenv

from langchain.chains. summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Load environment variables from .env file
load_dotenv()

file_path = 'example.pdf'

def summarize_pdf(file_path, ai='OpenAI'):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    
    if ai == 'OpenAI':
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    elif ai == "Claude":
        llm = ChatAnthropic(temperature=0, model_name="claude-3.5-haiku-latest")
    else:
        print("error")
        break


    chain = load_summarize_chain(llm=llm, chain_type='map_reduce')
    summary = chain.invoke(docs)

    return summary





if __name__ == '__main__':
    summary = summarize_pdf()
    print('Summary: %s' % summary)
