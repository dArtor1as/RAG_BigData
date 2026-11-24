"""Тести пайплайну запиту без справжніх LLM та індексу (фейковий ретривер і фейкова LLM)."""
import re
import unittest
from unittest import mock

import numpy as np

from src import generator, query
from src.citations import parse_citations, split_sentences, verify_citations

DOCS = [
    {"row_id": 0, "arxiv_id": "0704.0001", "title": "Sensor energy",
     "text": "Sensor energy. Neighboring sensor nodes share relaying to save battery energy."},
    {"row_id": 1, "arxiv_id": "0704.0002", "title": "Arabic speech",
     "text": "Arabic speech. An open source recognizer is adapted to continuous Arabic speech."},
    {"row_id": 2, "arxiv_id": "0704.0003", "title": "Graph coloring",
     "text": "Graph coloring. Random graph colorings change as more edges are added."},
]


def bow(texts):
    """Детерміновані «ембедінги»: мішок слів, захешований у 64 виміри і нормалізований."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            out[i, sum(map(ord, word)) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FakeRetriever:
    def encode(self, texts, side="query"):
        return bow(texts)

    def search_vector(self, vector, k):
        scores = bow([d["text"] for d in DOCS]) @ vector[0]
        order = np.argsort(-scores)[:k]
        return [{**DOCS[i], "score": float(scores[i])} for i in order]


class CitationParsing(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_citations("a [1][3] b"), [1, 3])
        self.assertEqual(parse_citations("a [1, 3]"), [1, 3])
        self.assertEqual(parse_citations("a [2-4]"), [2, 3, 4])
        self.assertEqual(parse_citations("no citations"), [])

    def test_citation_after_period_stays_with_sentence(self):
        parts = split_sentences("Sensors share relaying. [1] Arabic speech is recognized [2].")
        self.assertEqual(parts, ["Sensors share relaying [1].", "Arabic speech is recognized [2]."])
        self.assertEqual(split_sentences("- First point [1].\n- Second point [2]."),
                         ["First point [1].", "Second point [2]."])

    def test_statuses(self):
        text = ("Neighboring sensor nodes share relaying to save battery energy [1]. "
                "This sentence has no citation at all. "
                "Random graph colorings change with more edges [7]. "
                "Neighboring sensor nodes share relaying to save energy [2].")
        claims = verify_citations(text, DOCS, bow, threshold=0.6)
        self.assertEqual([c["status"] for c in claims], ["ok", "no_citation", "bad_index", "low_similarity"])


class Answer(unittest.TestCase):
    def setUp(self):
        fake = FakeRetriever()
        self.patches = [mock.patch.object(query, "get_retriever", return_value=fake),
                        mock.patch.object(generator, "get_retriever", return_value=fake)]
        for p in self.patches:
            p.start()
        self.thresholds = {"dense": 0.3, "rerank": 0.0}

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_prompt_numbers_sources_and_answer_is_verified(self):
        llm = mock.Mock(return_value="Neighboring sensor nodes share relaying to save battery energy [1].")
        with mock.patch.object(generator, "generate", llm):
            result = generator.answer("How do sensor nodes save battery energy?", k=2, model="m",
                                      backend="ollama", thresholds=self.thresholds)
        prompt = llm.call_args.args[0]
        self.assertIn("[1] (arXiv:0704.0001)", prompt)
        self.assertIn("[2] (arXiv:", prompt)
        self.assertEqual([s["n"] for s in result["sources"]], [1, 2])
        self.assertEqual(result["sources"][0]["url"], "https://arxiv.org/abs/0704.0001")
        self.assertFalse(result["refused"])
        self.assertEqual(result["claims"][0]["status"], "ok")
        self.assertIn("llm", result["timings"])

    def test_low_score_refuses_without_calling_llm(self):
        llm = mock.Mock()
        with mock.patch.object(generator, "generate", llm):
            result = generator.answer("Яка найкраща температура для випікання хліба?", backend="ollama",
                                      model="m", thresholds={"dense": 0.99, "rerank": 0.0})
        llm.assert_not_called()
        self.assertTrue(result["refused"])
        self.assertEqual(result["refusal_reason"], "low_score")
        self.assertEqual(result["answer"], generator.REFUSAL["uk"])

    def test_llm_not_found_becomes_refusal(self):
        with mock.patch.object(generator, "generate", return_value="NOT_FOUND"):
            result = generator.answer("How do sensor nodes save battery energy?", backend="ollama",
                                      model="m", thresholds=self.thresholds)
        self.assertTrue(result["refused"])
        self.assertEqual(result["refusal_reason"], "llm")

    def test_without_rag_no_sources(self):
        with mock.patch.object(generator, "generate", return_value="Plain answer.") as llm:
            result = generator.answer("What is BERT?", use_rag=False, backend="ollama", model="m")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["answer"], "Plain answer.")
        self.assertNotIn("Sources", llm.call_args.args[0])

    def test_hyde_uses_query_similarity_for_refusal(self):
        with mock.patch.object(query, "generate", return_value="Sensor nodes relay to save battery energy."), \
                mock.patch.object(generator, "generate", return_value="Nodes share relaying [1]."):
            result = generator.answer("How do sensor nodes save battery energy?", hyde=True, backend="ollama",
                                      model="m", thresholds=self.thresholds)
        self.assertEqual(result["hyde_passage"], "Sensor nodes relay to save battery energy.")
        expected = float(bow(["How do sensor nodes save battery energy?"])[0] @ bow([DOCS[0]["text"]])[0])
        self.assertAlmostEqual(result["relevance"], expected, places=5)
        self.assertIn("hyde", result["timings"])


if __name__ == "__main__":
    unittest.main()
