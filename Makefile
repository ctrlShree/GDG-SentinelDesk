.PHONY: install split train index test eval api web web-build ui
install:
	pip install -r requirements.txt
	cd frontend && npm install
split:
	python -m sentineldesk.data
train:
	python scripts/train_triage.py
index:
	python scripts/build_index.py
test:
	pytest -q
	cd frontend && npm run lint
eval:
	python scripts/evaluate.py --n 2000
api:
	uvicorn api:app --reload --port 8000
web:
	cd frontend && npm run dev
web-build:
	cd frontend && npm run build && npm start
ui:
	streamlit run app.py
