# HyperGraph Demo Notebooks

Interactive notebooks demonstrating [HyperGraph](https://github.com/gilad-rubin/hypergraph) — a Python framework for building AI/ML workflows using explicit graphs.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_basic_rag.ipynb` | Basic RAG pipeline: embed → retrieve → generate |
| `02_hierarchy.ipynb` | Composing graphs inside graphs with `.as_node()` |
| `03_map.ipynb` | Write for one item, scale to many with `.map()` |
| `04_cycles.ipynb` | Interactive chat loop with `InterruptNode` |

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Clone and install dependencies:
   ```bash
   git clone https://github.com/gilad-rubin/hypergraph-ds-leads.git
   cd hypergraph-ds-leads
   uv sync
   ```

3. Create a `.env` file with your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your key
   ```

4. Run the notebooks:
   ```bash
   uv run jupyter lab
   ```

## Requirements

- Python 3.12+
- OpenAI API key
