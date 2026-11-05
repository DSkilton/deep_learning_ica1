"""Static notebook compilation catches indentation and source-syntax mistakes."""
import json
from pathlib import Path


def test_notebook_code_cells_compile():
    for notebook in ("main.ipynb", "use_model.ipynb"):
        document = json.loads(Path(notebook).read_text(encoding="utf-8"))
        assert document["cells"], notebook
        for number, cell in enumerate(document["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = cell["source"]
            source = "".join(source) if isinstance(source, list) else source
            compile(source, f"{notebook}:cell{number}", "exec")
