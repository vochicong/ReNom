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

sphinx-versioning build -W 'v2.4.1' -w nothing -r v2.4.1 doc _build/html -- -D language='en'
sphinx-versioning build -W 'v2.4.1' -w nothing -r v2.4.1 doc _build/html/ja -- -D language='ja'
