Proyek dalam perkembangan. Cek berkala dlm folder V2
Progress track: Agentic Scraper & Querybot
Next: Implementasi context history chat ke querybot (cek condenser dr ALPHA1-2-3), scholarly utk scrape & simpulkan konsentrasi dosen, multi-summarizer & bureaucracy error-check
Dropped: auto clear cache utk FastEmbedEmbeddings & SentenceTransformers, publikasi per dosen (secara manual. cek "next")

Recommendation (failed experiment(s). use as you see fit [see ALPHA1-2-3.py]): 
  1. Peningkatan hasil menggunakan manual Embedding: Berhasil, namun max token 511 utk mxb, nomic, snowf, bge. JinaV2 bisa sampai 2047 sblm degradasi (cek testVector.ipynb)
  2. Limiter, Splitter, & Tokenizer: Multitude of errors
  3. Scholarly deep search & batch publication summarizer: IP blocked [suggestion: proxy & IP pool shuffle]
  4. Sentence Transformer & BERT: Hasil tidak bagus. Sama buruknya spt OllamaEmbeddings & selambat FastEmbedEmbeddings + cache numpuk