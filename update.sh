#!/bin/bash

echo "⚽ FIFA WC 2026 — Live Update Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate venv
source venv/bin/activate

# Step 1: Fetch latest results
echo "\n📡 Fetching latest match results..."
python src/updater.py

# Step 2: Re-run simulator to update win probabilities
echo "\n🎲 Re-running Monte Carlo simulator..."
python src/simulator.py

# Step 3: Push to GitHub
echo "\n🚀 Pushing to GitHub..."
git add data/processed/live_matches.csv data/processed/wc2026_predictions.json
git commit -m "live update - $(date '+%Y-%m-%d %H:%M')"
git push deploy main

echo "\n✅ Done! Dashboard will update in ~60 seconds."
echo "🌐 https://wc2026-oracle.streamlit.app/"