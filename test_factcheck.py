
import unittest
from unittest.mock import patch

import app as app_module


class FactCheckFeatureTests(unittest.TestCase):
    def test_search_google_fact_checks_missing_api_key(self):
        with patch.object(app_module, "GOOGLE_FACT_CHECK_API_KEY", ""):
            result = app_module.search_google_fact_checks("A headline claiming climate policy works.")

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["claims"], [])

    def test_search_google_fact_checks_matches(self):
        mock_response = type(
            "Resp",
            (),
            {
                "status_code": 200,
                "raise_for_status": lambda self: None,
                "json": lambda self: {"claims": [{"text": "The claim", "claimReview": [{"publisher": {"name": "PolitiFact"}, "textualRating": "Mostly False", "title": "Fact-check title", "url": "https://example.com/fact-check", "reviewDate": "2026-01-01"}]}]},
            },
        )()

        with patch.object(app_module, "GOOGLE_FACT_CHECK_API_KEY", "abc123"), patch.object(app_module.requests, "get", return_value=mock_response):
            result = app_module.search_google_fact_checks("A claim earlier in the article")

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["error_type"], "success")
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["publisher"], "PolitiFact")

    def test_search_google_fact_checks_no_matches(self):
        mock_response = type("Resp", (), {"status_code": 200, "raise_for_status": lambda self: None, "json": lambda self: {"claims": []}})()

        with patch.object(app_module, "GOOGLE_FACT_CHECK_API_KEY", "abc123"), patch.object(app_module.requests, "get", return_value=mock_response):
            result = app_module.search_google_fact_checks("A claim earlier in the article")

        self.assertEqual(result["status"], "no_matches")
        self.assertEqual(result["error_type"], "no_matches")
        self.assertEqual(result["claims"], [])

    def test_search_google_fact_checks_timeout(self):
        with patch.object(app_module, "GOOGLE_FACT_CHECK_API_KEY", "abc123"), patch.object(app_module.requests, "get", side_effect=app_module.requests.Timeout("timeout")):
            result = app_module.search_google_fact_checks("A claim earlier in the article")

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["claims"], [])

    def test_analyze_page_renders_external_fact_check_section(self):
        fake_result = {
            "label": "REAL",
            "confidence": 82.5,
            "confidence_level": "HIGH",
            "fake_prob": 18.0,
            "real_prob": 82.0,
            "model_used": "SVM",
            "decision_reason": "baseline",
            "net_score": 0,
            "uncertain_score": 0,
            "credibility_flags": [],
            "real_flags": [],
            "bert_ok": False,
            "bert_label": "N/A",
            "bert_fake": 0,
            "bert_real": 0,
            "bert_confidence": 0,
            "svm_label": "REAL",
            "svm_fake": 18.0,
            "svm_real": 82.0,
            "svm_confidence": 82.5,
            "keywords": [],
            "red_flags": {},
            "fact_checks": {
                "status": "matched",
                "claims": [{
                    "claim": "The claim being checked",
                    "publisher": "PolitiFact",
                    "rating": "Mostly False",
                    "title": "Fact-check title",
                    "url": "https://example.com/fact-check",
                    "review_date": "2026-01-01",
                }],
            },
        }

        with patch.object(app_module, "run_analysis", return_value=fake_result):
            app_module.app.config["WTF_CSRF_ENABLED"] = False
            client = app_module.app.test_client()
            response = client.post(
                "/analyze",
                data={"news_text": "This is a long enough text for analysis and review."},
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("External Fact-Check Evidence", body)
        self.assertIn("PolitiFact", body)


if __name__ == "__main__":
    unittest.main()
