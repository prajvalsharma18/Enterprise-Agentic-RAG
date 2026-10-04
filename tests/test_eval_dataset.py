import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluation.eval_dataset import EVAL_DATASET
from evaluation.generate_ground_truth_pdf import generate_ground_truth_pdf


class EvaluationDatasetTests(unittest.TestCase):
    def test_dataset_has_unique_ids_and_complete_traceability_fields(self):
        self.assertEqual(len(EVAL_DATASET), 30)
        self.assertEqual(len({item["id"] for item in EVAL_DATASET}), len(EVAL_DATASET))
        required = {
            "id", "question", "ground_truth", "source_document", "source_pages",
            "source_section", "category", "difficulty",
        }
        for item in EVAL_DATASET:
            self.assertTrue(required.issubset(item))
            self.assertTrue(item["question"].strip())
            self.assertTrue(item["ground_truth"].strip())
            self.assertIn(item["difficulty"], {"easy", "medium", "hard"})
            self.assertIn(
                item["category"],
                {
                    "exact_fact", "semantic_retrieval", "terminology_hybrid",
                    "multi_hop", "cross_document", "unsupported",
                },
            )

    def test_dataset_covers_documents_and_negative_questions(self):
        counts = Counter(item["category"] for item in EVAL_DATASET)
        self.assertEqual(counts["unsupported"], 3)
        self.assertEqual(counts["cross_document"], 2)
        self.assertEqual(
            {item["source_document"] for item in EVAL_DATASET if item["id"].startswith("enc-")},
            {"encryption_google_cloud.pdf"},
        )
        self.assertEqual(
            {item["source_document"] for item in EVAL_DATASET if item["id"].startswith("sec-")},
            {"google_cloud_security.pdf"},
        )
        self.assertEqual(
            {item["source_document"] for item in EVAL_DATASET if item["id"].startswith("ms-")},
            {"microsoft_annual_report_2025.pdf"},
        )

    def test_ground_truth_pdf_is_generated_from_dataset(self):
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ground_truth.pdf"
            with patch("evaluation.generate_ground_truth_pdf.OUTPUT_FILE", output):
                generated = generate_ground_truth_pdf()

            contents = generated.read_bytes()
            self.assertTrue(contents.startswith(b"%PDF-1.4"))
            self.assertEqual(contents.count(b"/Type /Page "), 15)
            self.assertIn(b"enc-01", contents)
            self.assertIn(b"neg-03", contents)


if __name__ == "__main__":
    unittest.main()
