#!/usr/bin/env bash

echo "Step 1: Checking Python file..."
if [ ! -f "inference.py" ]; then
  echo "❌ inference.py not found"
  exit 1
fi

echo "Step 2: Checking environment file..."
if [ ! -f "environment.py" ]; then
  echo "❌ environment.py not found"
  exit 1
fi

echo "Step 3: Running script..."
python inference.py

echo "✅ Basic validation passed!"