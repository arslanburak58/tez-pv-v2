.PHONY: help sync-data dataset features splits train-base train-meta optuna robustness baselines evaluate demo clean test lint

help:
	@echo "Pipeline komutları:"
	@echo "  make sync-data     - v1'den ham veriyi kopyala"
	@echo "  make dataset       - Veri seti oluştur (normalize, split, flags)"
	@echo "  make features      - pvlib fiziksel öznitelikleri üret"
	@echo "  make splits        - Zaman serisi cv split'lerini üret"
	@echo "  make train-base    - 9 base modeli eğit"
	@echo "  make train-meta    - QuantileRegressor meta-öğrenici eğit"
	@echo "  make optuna        - Hiperparametre optimizasyonu"
	@echo "  make robustness    - Sensör arıza testleri"
	@echo "  make baselines     - k-NN, SVM, LSTM, TFT"
	@echo "  make evaluate      - Final değerlendirme + holdout"
	@echo "  make demo          - Streamlit demo başlat"
	@echo "  make all           - Tüm pipeline (sırayla)"
	@echo ""
	@echo "Geliştirme:"
	@echo "  make test          - pytest"
	@echo "  make lint          - ruff + mypy"
	@echo "  make clean         - Çıktıları temizle"

sync-data:
	@echo "v1'den ham veri kopyalanıyor..."
	cp -r ~/Desktop/tez-pv/data/raw/dkasc/* data/raw/dkasc/ 2>/dev/null || echo "DKASC zaten kopyalı"
	cp -r ~/Desktop/tez-pv/data/raw/pvod/* data/raw/pvod/ 2>/dev/null || echo "PVOD zaten kopyalı"

dataset:
	python scripts/make_dataset.py

features:
	python scripts/build_features.py

splits:
	python scripts/split_dataset.py

train-base:
	python scripts/train_base_models.py

train-meta:
	python scripts/train_meta_learner.py

optuna:
	python scripts/run_optuna.py

robustness:
	python scripts/robustness_test.py

baselines:
	python scripts/train_baselines.py

evaluate:
	python scripts/evaluate.py

demo:
	streamlit run app/app.py

all: dataset features splits train-base train-meta optuna robustness baselines evaluate
	@echo "Pipeline tamamlandı."

test:
	pytest tests/ -v

lint:
	ruff check .
	mypy --ignore-missing-imports features/ models/ scripts/

clean:
	rm -rf data/processed/*.joblib data/processed/*.parquet
	rm -rf results/*.joblib results/*.parquet results/figures/*
	@echo "Çıktılar temizlendi."
