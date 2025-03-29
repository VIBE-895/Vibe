# Created by guxu at 3/29/25
from langchain_core.prompts import PromptTemplate

def get_stuffing_prompt():
    return PromptTemplate(
        template= """
        You are a summarization assistant. Read the following text carefully and output a structured summary in JSON format.

        Your task has four parts:
        1. Identify the setting — describe the type of situation (e.g., a course, meeting, or conversation). Use clear clues from the text to make a reasonable guess, but do not invent specific details.
        2. Determine the topic — write a short phrase summarizing the main subject of the document.
        3. Extract key terms — choose 3 to 8 important keywords or phrases that represent the core content.
        4. Write a combined summary — merge the summaries into a complete and well-structured paragraph, avoiding redundancy and preserving all important information.

        Rules:
        - Use complete sentences in the summary.
        - Avoid speculation or vague language.
        - The output must be a valid JSON object:

        Here is the text:

        {text}
        """,
        input_variables=["text"]
    )


def get_summarize_chunk_prompt():
    return PromptTemplate(
        template="""
        You are an assistant. Read the following text and write a short summary.

        The summary should:
        - Clearly cover the main idea and key details of the text.
        - Be written in complete sentences.
        - Be concise and avoid unnecessary repetition.

        The output must be in JSON format and contain only the summary.

        Text:
        {text}
        """,
        input_variables=["text"],
    )


def get_combine_chunk_prompt():
    return PromptTemplate(
        template="""
        You are a summarization assistant. You will be given a list of short summaries from different parts of a longer document.

        Your task is to:
        1. Identify the setting — describe the type of situation (e.g., a course, meeting, or conversation). Use clear clues from the text to make a reasonable guess, but do not invent specific details.
        2. Determine the topic — write a short phrase summarizing the main subject of the document.
        3. Extract key terms — choose 3 to 8 important keywords or phrases that represent the core content.
        4. Write a combined summary — merge the summaries into a complete and well-structured paragraph, avoiding redundancy and preserving all important information.

        The output must be in JSON format. Do not include any explanations, headers, or extra text.

        Here are the input summaries:
        {text}
        """,
        input_variables=["text"]
    )

def get_extract_query_prompt():
    return PromptTemplate(
        template= """
        You are an assistant. Your task is to help retrieve useful information from a document.

        Read the content below and generate several short search questions. 
        Each question should help identify an important part of the document (such as key decisions, tasks, problems, or plans). 

        Only generate a question if the content suggests that this part may exist but is not fully clear or may require more context. 
        Do not ask obvious or unnecessary questions. 
        If everything is already clear, do not ask about it.

        Rules:
        - Each question must be concise (less than 15 words).
        - Focus on different aspects of the document.
        - Do NOT explain or number the questions.
        - Only return the list of questions, one per line.

        Content:
        {text}
        
        Your summary should be formatted as a JSON object.
        """,
        input_variables = ["text"]
    )