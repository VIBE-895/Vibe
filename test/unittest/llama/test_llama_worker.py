# Created by guxu at 3/27/25
import shutil

from apps.llama.llama_worker import LlamaWorker
from apps import PROJECT_BASE_PATH
import os
import logging

MODEL_NAME = 'stablelm-zephyr:3b'
llama_worker = LlamaWorker(MODEL_NAME)

text = '''
    Ad sales boost Time Warner profit

    Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (£600m) for the three months to December, from $639m year-earlier.

    The firm, which is now one of the biggest investors in Google, benefited from sales of high-speed internet connections and higher advert sales. TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn. Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL.

    Time Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues. It hopes to increase subscribers by offering the online service free to TimeWarner internet customers and will try to sign up AOL's existing customers for high-speed broadband. TimeWarner also has to restate 2000 and 2003 results following a probe by the US Securities Exchange Commission (SEC), which is close to concluding.

    Time Warner's fourth quarter profits were slightly better than analysts' expectations. But its film division saw profits slump 27% to $284m, helped by box-office flops Alexander and Catwoman, a sharp contrast to year-earlier, when the third and final film in the Lord of the Rings trilogy boosted results. For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn. "Our financial performance was strong, meeting or exceeding all of our full-year objectives and greatly enhancing our flexibility," chairman and chief executive Richard Parsons said. For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins.

    TimeWarner is to restate its accounts as part of efforts to resolve an inquiry into AOL by US market regulators. It has already offered to pay $300m to settle charges, in a deal that is under review by the SEC. The company said it was unable to estimate the amount it needed to set aside for legal reserves, which it previously set at $500m. It intends to adjust the way it accounts for a deal with German music publisher Bertelsmann's purchase of a stake in AOL Europe, which it had reported as advertising revenue. It will now book the sale of its stake in AOL Europe as a loss on the value of that stake.
    '''

def test_load_pdf():
    pdf_path = os.path.join(PROJECT_BASE_PATH, 'test', 'data', 'pdf', 'meta_BQ_preparation.pdf')
    llama_worker.load_pdf(pdf_path)
    docs = llama_worker.documents
    assert len(docs) > 0 and docs[0].page_content

def test_load_image():
    image_path = os.path.join(PROJECT_BASE_PATH, 'test', 'data', 'image', 'img_1.png')
    llama_worker.load_image(image_path)
    docs = llama_worker.documents
    print(docs)

def test_summarize_stuffing():
    llama_worker.load_text(text)
    summary = llama_worker.summarize()
    print(type(summary))
    print(summary)

def test_summarize_map_reduce():
    llama_worker.load_text(text)
    summary = llama_worker.summarize(map_reduce=True)
    print(summary)

def test_extract_query():
    llama_worker.load_text(text)
    questions = llama_worker.extract_query()
    print(questions)

def test_summarize_with_rag():
    text = "I want to make good preparations for interviewing with Meta. I learn some tips for Meta's interview"
    llama_worker.load_text(text)
    pdf_path = os.path.join(PROJECT_BASE_PATH, 'test', 'data', 'pdf', 'meta_BQ_preparation.pdf')
    llama_worker.load_pdf(pdf_path)
    summary = llama_worker.summarize_with_rag()
    print(summary)

def test_search_and_answer():
    to_search = """
    To summarize long texts effectively using LLMs, break the text into smaller chunks, summarize each chunk, and then combine those summaries, possibly iteratively, until a final, concise summary is achieved. 
    Here's a more detailed explanation:
    1. Chunking the Text:
    Divide and Conquer: The first step is to divide the long text into smaller, manageable chunks that fit within the LLM's context window (the maximum amount of text the model can process at once). 
    Chunking Strategies:
    By Sections: If the text has natural sections (e.g., chapters, paragraphs), you can use those as chunk boundaries. 
    Character-Based: If there are no clear sections, you can divide the text into equal-sized chunks based on character count. 
    Overlap: To avoid losing context between chunks, consider overlapping chunks (where the last few characters of one chunk are repeated at the beginning of the next). 
    Tools: Libraries like LangChain offer tools for text splitting and chunking. 
    2. Summarizing Each Chunk:
    LLM Input:
    Feed each chunk of text to the LLM, along with a prompt instructing it to generate a concise summary.
    Prompt Engineering:
    Tailor the prompt to the specific task and desired output format (e.g., a few sentences, bullet points). 
    3. Combining Summaries:
    Iterative Summarization:
    Summarize the summaries of the chunks, repeating this process until you achieve a final summary that is both concise and informative. 
    MapReduce Approach:
    This strategy involves summarizing each chunk independently (the "map" phase) and then combining these summaries (the "reduce" phase). 
    Refine Strategy:
    Another approach is to iteratively refine the summary by feeding the current summary and the next chunk to the LLM, asking it to generate a refined summary. 
    4. Tools and Libraries:
    LangChain:
    LangChain is a popular framework for developing LLM applications and offers tools for text summarization, including chunking and summarization strategies. 
    Other Libraries:
    Other libraries and tools can also be used for LLM summarization, such as those that integrate with specific LLM models or APIs. 
    Text Summarization of Large Documents using LangChain - GitHub
    Overview. Text summarization is an NLP task that creates a concise and informative summary of a longer text. LLMs can be used to c...

    GitHub
    How to use LLMs: Summarize long documents - DEV Community
    May 1, 2024 — Luckily, there exists a technique that can get an LLM to summarize a document longer than its context window size. The ...

    DEV Community
    Summarize Large Documents or Text Using LLMs and ...
    Jul 16, 2024 — Summarize Large Documents or Text Using LLMs and LangChain. ... If the entire text fits within the LLM's context windo...

    Medium · 
    Ranjeet Tiwari | Senior Architect - AI | IITJ
    Show all
    Generative AI is experimental. For legal advice, consult a professional.
    """
    client_path = os.path.join(PROJECT_BASE_PATH, 'test', 'unittest', 'knowledgebase', 'persistent_clients')
    os.makedirs(client_path, exist_ok=True)
    from apps.knowledgebase.knowledge import Knowledge
    knowledge = Knowledge("test_path", "test")
    knowledge.add([to_search])
    llama_worker.knowledge_base = knowledge
    res = llama_worker.search_and_answer("How can I summarize long text with llm?")
    print(res)


    if os.path.exists(client_path):
        shutil.rmtree(client_path)


