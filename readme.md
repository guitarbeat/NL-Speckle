<!-- find . -name "*.py" -type f | while read file; do
    echo "Processing: $file"
    echo "Vulture output:"
    vulture "$file"
    echo "Pyflakes output:"
    pyflakes "$file"
    echo "-------------------"
done > code_analysis.txt 2>&1 -->