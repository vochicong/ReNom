# document will be created at _build/html

make clean -e SPHINXOPTS="-D language='ja'" html
python rewrite-tutorial.py ja
mv _build/html _build/ja

make -e SPHINXOPTS="-D language='en'" html
python rewrite-tutorial.py en
mv _build/ja _build/html/ja

python rewrite_path.py

rm -r _build/html/old-versions
rm -r _build/html/ja/old-versions
unzip old-versions-reference.zip;rm -rf __MACOSX
mv old-versions-reference _build/html/old-versions

unzip jsMath-3.6e.zip;rm -rf __MACOSX
mv jsMath-3.6e _build/html/_static/jsMath-3.6e
