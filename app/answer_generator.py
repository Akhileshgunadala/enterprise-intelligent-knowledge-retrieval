import json
import urllib.request
import urllib.error


class AnswerGenerator:

    def __init__(self):

        self.model = "qwen2.5:3b-instruct"
        self.url = "http://localhost:11434/api/generate"

        print(
            f"Answer generator initialized with local model: "
            f"{self.model}"
        )


    # ============================================================
    # BUILD CONTEXT
    # ============================================================

    def build_context(self, results):

        context_parts = []

        for i, result in enumerate(
            results,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {i}
Document: {result['document']}
Page: {result['page']}

Content:
{result['text']}
"""
            )

        return "\n".join(context_parts)


    # ============================================================
    # BUILD PROMPT
    # ============================================================

    def build_prompt(self, query, results):

        context = self.build_context(results)

        prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the enterprise
knowledge-base context provided below.

STRICT RULES:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context does not contain enough information,
   clearly say that the information was not found
   in the available enterprise documents.
4. Give a concise and professional answer.
5. Preserve exact numbers, limits, dates, and policy IDs.
6. Mention the relevant source document and page.
7. Do not claim that something is allowed or required
   unless the supplied context supports it.

USER QUESTION:
{query}

ENTERPRISE KNOWLEDGE-BASE CONTEXT:
{context}

Provide the answer using only the context above.
"""

        return prompt


    # ============================================================
    # NORMAL GENERATION
    # ============================================================

    def generate(self, query, results):

        if not results:

            return (
                "I could not find relevant information "
                "in the enterprise knowledge base."
            )

        prompt = self.build_prompt(
            query,
            results
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        data = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=120
            ) as response:

                response_data = json.loads(
                    response.read().decode("utf-8")
                )

            return response_data[
                "response"
            ].strip()

        except urllib.error.URLError as e:

            return (
                "Unable to connect to the local Ollama server. "
                "Make sure Ollama is running.\n\n"
                f"Technical details: {e}"
            )

        except Exception as e:

            return (
                "An error occurred while generating the answer.\n\n"
                f"Technical details: {e}"
            )


    # ============================================================
    # STREAMING GENERATION
    # ============================================================

    def generate_stream(
        self,
        query,
        results
    ):

        if not results:

            yield (
                "I could not find relevant information "
                "in the enterprise knowledge base."
            )

            return


        prompt = self.build_prompt(
            query,
            results
        )


        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1
            }
        }


        data = json.dumps(
            payload
        ).encode("utf-8")


        request = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=120
            ) as response:

                # Ollama sends one JSON object per line
                for line in response:

                    line = line.decode(
                        "utf-8"
                    ).strip()


                    if not line:
                        continue


                    try:

                        response_data = json.loads(
                            line
                        )

                    except json.JSONDecodeError:

                        continue


                    token = response_data.get(
                        "response",
                        ""
                    )


                    if token:

                        yield token


                    # Ollama marks completion with done=True
                    if response_data.get(
                        "done",
                        False
                    ):

                        break


        except urllib.error.URLError as e:

            yield (
                "\n\nUnable to connect to the local "
                "Ollama server. Make sure Ollama is running.\n\n"
                f"Technical details: {e}"
            )


        except Exception as e:

            yield (
                "\n\nAn error occurred while generating "
                "the answer.\n\n"
                f"Technical details: {e}"
            )