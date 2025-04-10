# Created by guxu at 10/24/24
from ..llama.llama_worker import LlamaWorker

class TextSummaryService:
    def __init__(self, model_name="stablelm-zephyr:3b"):
        self.worker = LlamaWorker(model_name)

    def summarize_text(self, text):
        self.worker.load_text(text)
        return self.worker.summarize()

    def intelligent_query(self, query):
        supportive_doc, answer = self.worker.search_and_answer(query)
        return supportive_doc, answer