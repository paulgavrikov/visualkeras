#!/bin/bash
# Simple documentation build script for local development

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCS_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$( cd "$DOCS_DIR/.." && pwd )"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Build HTML documentation
echo "Building documentation..."
cd "$DOCS_DIR"
sphinx-build -b html -d build/doctrees source build/html

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Documentation built successfully!"
    echo "  View it at: file://$DOCS_DIR/build/html/index.html"
    echo ""
    echo "To serve locally, run:"
    echo "  cd $DOCS_DIR && python -m http.server 8000 --directory build/html"
else
    echo "✗ Documentation build failed!"
    exit 1
fi
