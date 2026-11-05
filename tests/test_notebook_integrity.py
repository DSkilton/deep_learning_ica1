"""Fast structural tests: no TensorFlow training required."""
import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class NotebookTests(unittest.TestCase):
    def test_notebooks_are_valid_python_cells(self):
        for name in ("main.ipynb", "use_model.ipynb"):
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
                for i, cell in enumerate(notebook["cells"]):
                    if cell.get("cell_type") == "code":
                        source = "".join(cell.get("source", []))
                        with self.subTest(cell=i):
                            ast.parse(source)

    def test_transformer_mask_and_training_call(self):
        nb = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"
        )
        self.assertIn("GlobalAveragePooling1D()(X, mask=", source)
        self.assertIn("model, history, macro_f1 = train_one_config(", source)
        self.assertIn("BPE_PREFIX = \"m_bpe_v2\"", source)

    def test_cleanup_script_is_syntactically_valid_and_opt_in(self):
        source = (ROOT / "scripts/top10.py").read_text(encoding="utf-8")
        compile(source, "scripts/top10.py", "exec")
        self.assertIn('parser.add_argument("--apply"', source)
        self.assertIn("if APPLY_DELETIONS:", source)

if __name__ == "__main__":
    unittest.main()
