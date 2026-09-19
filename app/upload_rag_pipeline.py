from upload_enterprise_retriever import UploadEnterpriseRetriever
from answer_generator import AnswerGenerator


class UploadRAGPipeline:

    def __init__(self):
        print("\n===================================")
        print("INITIALIZING UPLOAD RAG PIPELINE")
        print("===================================\n")

        self.retriever = UploadEnterpriseRetriever()
        self.answer_generator = AnswerGenerator()

        print("\nUpload RAG pipeline ready.")

    # -----------------------------------------
    # NORMAL / NON-STREAMING ASK
    # -----------------------------------------
    def ask(self, query, retrieval_k=10, top_k=3):

        print("\n===================================")
        print("RAG QUERY")
        print("===================================\n")

        print(f"Question: {query}")

        results = self.retriever.search(
            query,
            retrieval_k=retrieval_k,
            top_k=top_k
        )

        answer = self.answer_generator.generate(
            query,
            results
        )

        return {
            "query": query,
            "answer": answer,
            "sources": results
        }

    # -----------------------------------------
    # STREAMING ASK
    # -----------------------------------------
    def ask_stream(self, query, retrieval_k=10, top_k=3):

        print("\n===================================")
        print("STREAMING RAG QUERY")
        print("===================================\n")

        print(f"Question: {query}")

        # Step 1: Retrieve relevant documents
        results = self.retriever.search(
            query,
            retrieval_k=retrieval_k,
            top_k=top_k
        )

        print("\nGenerating answer with local LLM...")

        # Step 2: Generate answer token by token
        answer_chunks = []

        for chunk in self.answer_generator.generate_stream(
            query,
            results
        ):

            answer_chunks.append(chunk)

            # Send token to Streamlit
            yield {
                "type": "token",
                "content": chunk
            }

        # Step 3: Combine complete answer
        final_answer = "".join(answer_chunks).strip()

        print("\nAnswer generation completed.")

        # Step 4: Send final result + sources
        yield {
            "type": "complete",
            "answer": final_answer,
            "sources": results
        }


# -----------------------------------------
# TEST
# -----------------------------------------
if __name__ == "__main__":

    pipeline = UploadRAGPipeline()

    result = pipeline.ask(
        "How many annual leave days do employees receive?"
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:
        print(
            f"- {source['document']} "
            f"(Page {source['page']})"
        )