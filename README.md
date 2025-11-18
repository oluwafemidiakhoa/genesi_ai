GENESIS-AI
A unified, open-source platform for AI-driven protein design, analysis, and protocol generation.

🚀 Features
Multi-Agent Design Studio
Orchestrates planning, research, design, simulation, and validation agents to invent novel protein candidates.

Protein Designer
Upload or fetch PDB/mmCIF structures, run pocket analysis (PISA, DoGSite), visualize 3D models, generate mutagenesis suggestions, and export.

Protocol Generator
AI-powered stepwise experimental protocols via Google Gemini (or OpenAI fallback) with integrated drug safety (OpenFDA), clinical trials lookup, and export.

PDF Report Generator
Compile summaries, AI analyses, and predicted pockets into polished PDF reports.

Literature AI Panel
PubMed search + abstract fetch + AI summarization pipelines.

Knowledge Explorer
Build and visualize biomedical knowledge graphs in Neo4j using Gemini extraction + custom KG builder/viewer.

AlphaFold DB Lookup
Search by UniProt ID, preview model metadata, and download predicted structures.

Extensible Agent Framework
Meta-agent coordinates specialized sub-agents:

planning_agent.py

research_agent.py

design_agent.py

simulation_agent.py

validation_agent.py

protocol_agent.py

reflection_agent.py

Configurable Workflows
Toggle dev mode, docking/admet/toxicity thresholds, design batch size, optimization focus, and more.

📁 Repository Layout
bash
Copy
Edit
.
├── .env                     # Environment variables & API keys
├── requirements.txt         # Python dependencies
├── main.py                  # Streamlit entrypoint
├── genesis_ai.py            # Core engine and agent orchestrator
├── ui_components.py         # Legacy UI stubs (migrating to `ui/` folder)
├── knowledge_explorer/      # Streamlit knowledge-graph app
│   └── knowledge_explorer.py
├── agents/                  # Multi-agent architecture
│   ├── meta_agent.py
│   ├── planning_agent.py
│   ├── research_agent.py
│   ├── design_agent.py
│   ├── simulation_agent.py
│   ├── validation_agent.py
│   ├── protocol_agent.py
│   └── reflection_agent.py
├── knowledge/               # KG builder, Neo4j client, external integrations
│   ├── kg_builder.py
│   ├── knowledge_graph.py
│   ├── umls_integration.py
│   ├── bioportal_integration.py
│   ├── pubmed_client.py
│   └── chembl_client.py
├── models/                  # ML and neuro-symbolic models
│   ├── hybrid_model.py
│   ├── symbolic_engine.py
│   ├── neuro_symbolic.py
│   └── generative_smiles.py
├── simulation/              # Simulators and external tool integrations
│   ├── bioc_simulator.py
│   ├── copasi_integration.py
│   └── alphafold_client.py
└── utils/                   # Shared utilities
    ├── api_utils.py
    ├── api_helpers.py
    └── visualization.py
⚙️ Installation
Clone the repo

bash
Copy
Edit
git clone https://github.com/your-org/genesis-ai.git
cd genesis-ai
Create & activate a virtual environment

bash
Copy
Edit
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Configure environment variables
Copy .env.example to .env and fill in your API keys:

ini
Copy
Edit
UMLS_API_KEY=
NCBI_API_KEY=
BIOPORTAL_API_KEY=
OPENAI_API_KEY=
HF_TOKEN=
NCBI_EMAIL=you@example.com
ELEVEN_LABS_API_KEY=
GEMINI_API_KEY=
NEO4J_PASSWORD=
JBEI_ICE_USER=
JBEI_ICE_PWD=
HF_PROTEIN_MODEL=nvidia/esm2_t6_8M_UR50D
RCSB_INCLUDE_CSM=true
RCSB_ROWS=200

HTTP_TIMEOUT=20
HTTP_MAX_RETRIES=3
HTTP_BACKOFF_FACTOR=1.5
HTTP_USER_AGENT=genesis-ai/0.1

GENESIS_DEV_MODE=1
VALIDATION_DOCKING_MAX=-7.8
VALIDATION_ADMET_MIN=0.58
VALIDATION_TOX_MAX=0.40

NUM_DESIGNS=3
OPTIMIZATION_FOCUS=potency
MAX_MW=550
🚀 Running the App
bash
Copy
Edit
streamlit run main.py
Open your browser to http://localhost:8501.

Use the sidebar or top tabs to navigate features.

🧠 Knowledge Graph
Neo4j Backend: set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env.

Build & Visualize: enter a topic, click Build & Visualize, then explore the interactive graph.

🛠️ Development Tips
Dev Mode: set GENESIS_DEV_MODE=1 for extra logging and hot-reload.

Agent Debugging: each sub-agent logs to genesis-ai.log at INFO or DEBUG level.

Extending:

Add new agents in agents/.

Register new KG integrations in knowledge/.

Drop new protocol templates into helpers/generate_protocol.py.

🎓 Citation
If you use GENESIS-AI in your research or projects, please cite:

GENESIS-AI: “A Unified AI Platform for Protein Design and Analysis,” Foundry AI, 2025.

❤️ Contributing
Fork the repo

Create a feature branch

Submit a pull request

Please follow our Code of Conduct and Contributing Guidelines.

📄 License
MIT License © 2025 Foundry AI.
See LICENSE for full details."# genesi_ai" 
