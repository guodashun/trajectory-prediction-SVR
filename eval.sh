#!/bin/bash
load_path="./npz_502/"
save_path="./test_data/"
files=$(ls $load_path)
for file in $files
do 
    # read input
    # # echo $input
    # if [ $input == "n" ]
    # then
    #     rm ./test_data/*
    #     cd $load_path
    #     echo $file
    #     cp $file ../test_data/
    #     cd ..
    #     python eval.py
    # fi
    rm ./test_data/*
    cd $load_path
    echo $file
    cp $file ../test_data/
    cd ..
    python eval.py
done
