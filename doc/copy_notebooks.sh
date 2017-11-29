
path2ReNomNotebook="../../ReNomNotebook"
for file in ${path2ReNomNotebook}/notebooks/tutorial*;
do
    filename=`basename ${file}`
    if [ ! -e rsts/${filename}.ipynb ]; then
        echo ${filename} ${file}
        ln -s ../${file}/notebook.ipynb rsts/${filename}.ipynb
    fi
done
