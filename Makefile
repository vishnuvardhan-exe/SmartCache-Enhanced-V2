.PHONY: help install test demo clean docker-build docker-run example benchmark

help:
	@echo "SmartCache - Makefile Commands"
	@echo ""
	@echo "  make install     - Install dependencies in virtual environment"
	@echo "  make test        - Run unit tests"
	@echo "  make demo        - Start Streamlit demo (requires dependencies)"
	@echo "  make example     - Run example script (quick demo)"
	@echo "  make benchmark   - Run performance benchmark"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run with Docker Compose"
	@echo "  make clean       - Remove build artifacts and caches"
	@echo "  make package     - Create distributable wheel"

install:
	python3 -m venv venv
	. venv/bin/activate && pip install -U pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .
	@echo ""
	@echo "✅ Installation complete! Activate with: source venv/bin/activate"

test:
	. venv/bin/activate && python -m pytest tests/ -v

demo:
	. venv/bin/activate && streamlit run demo/app.py

example:
	. venv/bin/activate && python example.py

benchmark:
	. venv/bin/activate && python benchmark.py

docker-build:
	docker build -t smartcache:latest .

docker-run:
	docker-compose up -d
	@echo "✅ SmartCache demo running at http://localhost:8501"
	@echo "   Redis available at localhost:6379"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov dist build *.egg-info
	rm -rf smartcache_demo_data example_cache_data
	rm -rf venv 2>/dev/null || true
	@echo "✅ Clean complete"

package:
	. venv/bin/activate && python -m build
	@echo "✅ Package built in dist/"
