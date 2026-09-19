from enterprise_retriever import EnterpriseRetriever
from answer_generator import AnswerGenerator


class EnterpriseRAG:

    def __init__(self):

        print("=" * 70)
        print("INITIALIZING ENTERPRISE RAG SYSTEM")
        print("=" * 70)

        self.retriever = EnterpriseRetriever()
        self.generator = AnswerGenerator()

        print("\nEnterprise RAG system ready.")

    def ask(
        self,
        query,
        retrieval_k=10,
        top_k=3
    ):

        # Retrieval
        results = self.retriever.search(
            query,
            retrieval_k=retrieval_k,
            top_k=top_k
        )

        # Generation
        answer = self.generator.generate(
            query,
            results
        )

        return {
            "query": query,
            "answer": answer,
            "sources": results
        }


def run_test():

    print("\n")
    print("#" * 70)
    print("# ENTERPRISE RAG END-TO-END TEST")
    print("#" * 70)

    rag = EnterpriseRAG()

    query = "How many work from home days are allowed?"

    result = rag.ask(query)

    print("\n")
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(result["query"])

    print("\n")
    print("=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n")
    print("=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for rank, source in enumerate(
        result["sources"],
        start=1
    ):

        print(
            f"\nSource {rank}: "
            f"{source['document']} "
            f"(Page {source['page']})"
        )


if __name__ == "__main__":
    run_test()