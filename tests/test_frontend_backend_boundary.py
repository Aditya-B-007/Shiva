import ast
import unittest
from pathlib import Path

from src.contracts import block_schemas
from src.user.frontend.workflow_builder import build_workflow_request


ROOT = Path(__file__).resolve().parents[1]


class FrontendBackendBoundaryTests(unittest.TestCase):
    def test_workflow_builder_uses_contract_payload_shape(self):
        definitions = {
            schema.block_type: schema.to_json()
            for schema in block_schemas()
        }
        blocks = {
            "prompt_1": {
                "id": "prompt_1",
                "type": "prompt",
                "name": "Prompt Input",
                "arguments": {"device": "user_prompt", "capture_mode": "text"},
                "condition": {},
            },
            "output_1": {
                "id": "output_1",
                "type": "shiva_output",
                "name": "Shiva Output",
                "arguments": {"output_format": "text"},
                "condition": {},
            },
        }
        edges = [{"id": "edge_1", "source_id": "prompt_1", "target_id": "output_1"}]

        payload = build_workflow_request(blocks, edges, definitions, "Summarize input").to_json()

        self.assertEqual(payload["query"], "Summarize input")
        self.assertEqual(payload["output_format"], "text")
        self.assertEqual(payload["blocks"], [{"device": "user_prompt", "arguments": {"device": "user_prompt", "capture_mode": "text"}}])
        self.assertIn("custom_graph", payload["metadata"])

    def test_contract_schemas_are_frontend_safe(self):
        schemas = [schema.to_json() for schema in block_schemas()]
        self.assertTrue(any(schema["block_type"] == "prompt" for schema in schemas))
        for schema in schemas:
            self.assertIn("block_type", schema)
            self.assertIn("category", schema)
            self.assertIn("default_arguments", schema)
            self.assertIn("fields", schema)

    def test_frontend_does_not_import_backend_internals(self):
        forbidden_prefixes = (
            "src.orchestrator",
            "src.body",
            "src.brain",
            "src.swarm",
        )
        frontend_files = (ROOT / "src" / "user" / "frontend").glob("*.py")
        violations = []

        for file_path in frontend_files:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        if module and module.startswith(forbidden_prefixes):
                            violations.append((file_path.name, module))
                    continue

                if module and module.startswith(forbidden_prefixes):
                    violations.append((file_path.name, module))

        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
