# document will be created at _build/html

sphinx-versioning build -w doc_v2.4.1 -W nothing -r doc_v2.4.1 doc _build/html -- -D language='en'
sphinx-versioning build -w doc_v2.4.1 -W nothing -r doc_v2.4.1 doc _build/html/ja -- -D language='ja'

# sphinx-versioning build -r v2.4.1d doc _build/html/en -- -D language='en'
# mv _build/html/* _build/html/en

# unzip jsMath-3.6e.zip;
# rm -rf __MACOSX
# mv jsMath-3.6e _build/html/_static/jsMath-3.6e
