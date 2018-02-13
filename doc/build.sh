# document will be created at _build/html

DIR_STATIC="_static/jsMath-3.6e"
URL_MATH="https://jaist.dl.sourceforge.net/project/jsmath/jsMath/3.6e/jsMath-3.6e.zip"
ZIP_NAME="jsMath-3.6e.zip"

if [ -d ${DIR_STATIC} ]; then
    echo "jsMath exists."
else
    echo "Downloading jsMath"
    wget ${URL_MATH} 
    unzip -d _static ${ZIP_NAME} 
    rm ${ZIP_NAME}
fi

sphinx-versioning build -w doc_v2.4.1 -W nothing -r doc_v2.4.1 doc _build/html/ja -- -D language='ja'
sphinx-versioning build -w doc_v2.4.1 -W nothing -r doc_v2.4.1 doc _build/html/en -- -D language='en'

# mv _build/html/en/* _build/html/
# rmdir _build/html/en

# sphinx-versioning build -r v2.4.1d doc _build/html/en -- -D language='en'
# mv _build/html/* _build/html/en

# unzip jsMath-3.6e.zip;
# rm -rf __MACOSX
# mv jsMath-3.6e _build/html/_static/jsMath-3.6e
