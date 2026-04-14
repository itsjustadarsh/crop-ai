"""Convert crop_benchmark_v2.py → crop_benchmark_v2.ipynb using nbformat."""
import nbformat as nbf
from pathlib import Path

src = Path(__file__).parent / "crop_benchmark_v2.py"
out = Path(__file__).parent / "crop_benchmark_v2.ipynb"

lines = src.read_text().splitlines()

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.14.2"}

cells = []
current_type = None
current_lines = []

def flush():
    if not current_lines:
        return
    text = "\n".join(current_lines).strip()
    if not text:
        return
    if current_type == "markdown":
        # strip leading "# " from each line
        md_lines = []
        for ln in text.splitlines():
            if ln.startswith("# "):
                md_lines.append(ln[2:])
            elif ln.startswith("#"):
                md_lines.append(ln[1:])
            else:
                md_lines.append(ln)
        cells.append(nbf.v4.new_markdown_cell("\n".join(md_lines)))
    else:
        cells.append(nbf.v4.new_code_cell(text))

for line in lines:
    stripped = line.strip()
    if stripped.startswith("# %% [markdown]"):
        flush()
        current_type = "markdown"
        current_lines = []
    elif stripped.startswith("# %%"):
        flush()
        current_type = "code"
        current_lines = []
    else:
        if current_type is not None:
            current_lines.append(line)

flush()
nb.cells = cells

with open(out, "w") as f:
    nbf.write(nb, f)

print(f"✅ Notebook written to {out}  ({len(cells)} cells)")
