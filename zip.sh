#!/bin/sh

files=()
for file in $(cat zip_files.txt);do
    files+="${file} " 
done
echo $files

zip -r streamlit-rag-chatbot $files