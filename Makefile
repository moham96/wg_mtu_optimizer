.PHONY: install test clean lint format

install:
	@echo "Installing dependencies..."
	pip3 install -r requirements.txt
	@echo "Making script executable..."
	chmod +x wg_mtu_finder.py

test:
	@echo "Running basic tests..."
	python3 -c "import pandas, matplotlib, seaborn, numpy; print('✓ All Python dependencies available')"
	@which iperf3 > /dev/null && echo "✓ iperf3 available" || echo "✗ iperf3 not found"
	@which ping > /dev/null && echo "✓ ping available" || echo "✗ ping not found"  
	@which ip > /dev/null && echo "✓ ip command available" || echo "✗ ip command not found"

clean:
	@echo "Cleaning up output directories..."
	rm -rf wg_mtu_results*/
	rm -f *.log

lint:
	@echo "Linting Python code..."
	python3 -m py_compile wg_mtu_finder.py

format:
	@echo "Code formatting not implemented - use black or autopep8 if needed"

help:
	@echo "Available targets:"
	@echo "  install - Install dependencies"
	@echo "  test    - Test dependencies"
	@echo "  clean   - Clean output directories"
	@echo "  lint    - Check Python syntax"
	@echo "  help    - Show this help"
