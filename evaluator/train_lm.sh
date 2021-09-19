# /bin/bash
cur_path=$(cd `dirname $0`; pwd)

if [ ! -d "$cur_path/kenlm" ]; then
        mkdir $cur_path/kenlm
fi

for file in $(ls $(dirname $cur_path)/data/$1/)
do
    if [[ $file == train.* ]]; then
        style=${file##*.}
        cat $(dirname $cur_path)/data/$1/$file | ${kenlm}lmplz -o 5 --verbose > $1.$style.arpa
        ${kenlm}build_binary $1.$style.arpa $cur_path/kenlm/$1.$style.bin
        rm -f $1.$style.arpa
    fi
done