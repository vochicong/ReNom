# document will be created at _build/html

DIR_STATIC="_static/jsMath-3.6e"

if [ -d ${DIR_STATIC} ]; then
    echo "jsMath exists."
else
    echo "Downloading jsMath"
    unzip -d _static ../../jsMath-3.6e.zip
fi

rm -rf _build/html

sphinx-versioning build -W 'v2.4.1' -w 'aaa' -r 'v2.4.1' doc _build/html -- -D language='en'
python rewrite_tutorial.py en

mv _build/html _build/en

sphinx-versioning build -W 'v2.4.1' -w 'aaa' -r 'v2.4.1' doc _build/html -- -D language='ja'
python rewrite_tutorial.py ja
for f in `find _build -type d | grep 'static' | grep -v 'static/'`; do
    cp -r _static/* ${f}
done
mv _build/html _build/ja

mv _build/en _build/html
mv _build/ja _build/html/ja
