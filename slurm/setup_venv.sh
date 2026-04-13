#!/bin/bash
# ============================================================
# One-time setup: installs Python 3.12 + dependencies for
# running scripts on the RIS virtual desktop / login node.
#
# Run from the project root:
#   bash slurm/setup_venv.sh
# ============================================================

set -e

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"
PY_DIR="$PROJECT_DIR/python"
PY_VERSION="3.12.8"

echo "Project root: $PROJECT_DIR"

# Download and install standalone Python if not already present
if [ ! -f "$PY_DIR/bin/python3" ]; then
    echo "Downloading Python $PY_VERSION..."
    mkdir -p "$PY_DIR"
    curl -sSL "https://www.python.org/ftp/python/$PY_VERSION/Python-$PY_VERSION.tgz" -o /tmp/python.tgz
    echo "Extracting and building (this takes a few minutes)..."
    cd /tmp
    tar xzf python.tgz
    cd "Python-$PY_VERSION"
    ./configure --prefix="$PY_DIR" --enable-optimizations --with-ensurepip=install 2>&1 | tail -1
    make -j$(nproc) 2>&1 | tail -1
    make install 2>&1 | tail -1
    cd "$PROJECT_DIR"
    rm -rf /tmp/python.tgz /tmp/Python-$PY_VERSION
    echo "Python installed to $PY_DIR"
else
    echo "Python already installed at $PY_DIR"
fi

# Create venv
echo "Creating venv..."
"$PY_DIR/bin/python3" -m venv "$PROJECT_DIR/venv"
source "$PROJECT_DIR/venv/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "To activate in future sessions:"
echo "  source $PROJECT_DIR/venv/bin/activate"
echo ""
echo "Then run scripts:"
echo "  cd $PROJECT_DIR"
echo "  python scripts/01_crop.py"
echo "========================================="
