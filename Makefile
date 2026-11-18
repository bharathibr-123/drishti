# Shortcuts. On Windows without make, run the commands after each colon directly.
.PHONY: setup index run test check eval

setup:   ## install dependencies, download weights and demo videos
	pip install -r requirements.txt && python scripts/setup_demo.py
index:   ## build the search index from demo/cameras.json
	python -m drishti.ingest demo/cameras.json
run:     ## start the web app on http://localhost:8000
	uvicorn app.server:app --port 8000
test:    ## unit tests
	pytest -q
check:   ## lint + strict type-check + tests (what CI runs)
	ruff check . && ruff format --check . && mypy && pytest -q
eval:    ## benchmark + ablation -> eval/RESULTS.md
	python -m drishti.evaluate
