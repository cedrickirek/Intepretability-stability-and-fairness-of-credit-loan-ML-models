#!/bin/bash
set -euo pipefail

# Change to project root (this script lives in Interpretability_project/scripts)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR/.."

echo "Exporting notebooks to PDF via nbconvert (webpdf)..."

NOTEBOOKS=(
  "Group_9_Interpretability_project.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
  if [ -f "$nb" ]; then
    echo "- Exporting $nb"
    if jupyter nbconvert --to webpdf "$nb"; then
      echo "  -> PDF created"
    else
      echo "  -> webpdf export failed. Attempting HTML fallback..."
      jupyter nbconvert --to html "$nb" && echo "  -> HTML created as fallback"
      echo "  Tip: If webpdf fails, ensure 'nbconvert[webpdf]' and 'playwright' are installed, then run:"
      echo "       playwright install chromium"
    fi
  else
    echo "- Skipping $nb (not found)"
  fi
done

echo "Done. Outputs saved next to the notebooks."

