# Dense Retrieval Papers


> A collection of papers related to dense retrieval. 
>
> The arrangement of papers refers to our survey [**"Dense Text Retrieval based on Pretrained Language Models: A Survey"**](https://arxiv.org/abs/2211.14876).

If you find our survey useful for your research, please cite the following paper:
```
@article{DRSurvey,
    title={Dense Text Retrieval based on Pretrained Language Models: A Survey
},
    author={Wayne Xin Zhao, Jing Liu, Ruiyang Ren, Ji-Rong Wen},
    year={2022},
    journal={arXiv preprint arXiv:2211.14876}
}
```


## Table of Contents

- [Survey paper](#survey-paper)
- [Architecture](#architecture) 
- [Training](#training)
  - [Formulation](#formulation)
  - [Negative Selection](#negative-selection)
  - [Data Augmentation](#data-augmentation)
  - [Pre-training](#pre-training)
- [Indexing](#indexing)
- [Interation with Re-ranking](#interation-with-re-ranking)
- [Advanced Topics](#advanced-topics)
  - [Zero-shot Dense Retrieval](#zero-shot-dense-retrieval)
  - [Improving the Robustness to Query Variations](#improving-the-robustness-to-query-variations)
  - [Generative Text Retrieval](#generative-text-retrieval)
  - [Retrieval-Augmented Language Model](#retrieval-augmented-language-model)
- [Applications](#applications)
  - [Information Retrieval Applications](#information-retrieval-applications)
  - [Natural Language Processing Applications](#natural-language-processing-applications)
  - [Industrial Practice](#industrial-practice)
- [Datasets](#datasets)
- [Libraries](#libraries)

 
## Survey Paper
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Pretrained Transformers for Text Ranking: BERT and Beyond.](https://arxiv.org/pdf/2010.06467.pdf) | Jimmy Lin et al. | Synthesis HLT 2021 | NA |
| [Semantic Models for the First-stage Retrieval: A Comprehensive Review.](https://arxiv.org/pdf/2103.04831.pdf) | 	Yinqiong Cai et al. | Arxiv 2021 | NA |
| [Pre-training Methods in Information Retrieval.](https://arxiv.org/pdf/2111.13853) | Yixing Fan et al. | Arxiv 2021 | NA |
| [A Deep Look into Neural Ranking Models for Information Retrieval.](https://arxiv.org/pdf/1903.06902) | Jiafeng Guo et al. | Inf. Process. Manag. 2020 | NA |
|[Lecture Notes on Neural Information Retrieval.](https://arxiv.org/pdf/2207.13443.pdf) | Nicola Tonellotto. | Arxiv 2022 | NA
|[Low-Resource Dense Retrieval for Open-Domain Question Answering: A Comprehensive Survey.](https://arxiv.org/pdf/2208.03197.pdf) | Xiaoyu Shen et al. | Arxiv 2022 | NA


## Architecture

| **Paper** | **Author** | **Venue**  | **Code** |
| --- | --- | --- | --- |
| [Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring.](https://arxiv.org/pdf/1905.01969.pdf) | Samuel Humeau et al. | ICLR 2020 | [Python](https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder) |
| [Sparse, Dense, and Attentional Representations for Text Retrieval.](https://arxiv.org/pdf/2005.00181.pdf) | Yi Luan et al. | <div style="width: 150pt">TACL 2021 | [Python](https://github.com/google-research/language/tree/master/language/multivec) |
| [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.](https://arxiv.org/pdf/2004.12832.pdf) | Omar Khattab et al. | SIGIR 2020 | [Python](https://github.com/stanford-futuredata/ColBERT) |
| [Query Embedding Pruning for Dense Retrieval.](https://arxiv.org/pdf/2108.10341) | Nicola Tonellotto et al. | CIKM 2021 | [Python](https://github.com/terrierteam/pyterrier_colbert) |
| [Context-Aware Term Weighting For First Stage Passage Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3397271.3401204) | Zhuyun Dai et al. | SIGIR 2020 | [Python](https://github.com/AdeDZY/DeepCT) |
| [Context-Aware Document Term Weighting for Ad-Hoc Search.](https://dl.acm.org/doi/pdf/10.1145/3366423.3380258) | Zhuyun Dai et al. | WWW 2020 | [Python](https://github.com/AdeDZY/DeepCT/tree/master/HDCT) |
| [DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding.](https://arxiv.org/pdf/2002.12591.pdf) | Yuyu Zhang et al. | SIGIR 2020 | NA |
| [Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index.](https://arxiv.org/pdf/1906.05807.pdf) | Minjoon Seo et al. | ACL 2019 | [Python](https://github.com/uwnlp/denspi) |
| [Learning Dense Representations of Phrases at Scale.](https://arxiv.org/pdf/2012.12624.pdf) | Jinhyuk Lee et al. | ACL 2021 | [Python](https://github.com/jhyuklee/DensePhrases) |</div>
| [Phrase Retrieval Learns Passage Retrieval, Too. ](https://arxiv.org/pdf/2109.08133.pdf) | Jinhyuk Lee et al. | <div style="width: 150pt">EMNLP 2021</div> | [Python](https://github.com/princeton-nlp/DensePhrases) |
| [Dense Hierarchical Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2110.15439) | Ye Liu et al. | EMNLP 2021 | [Python](https://github.com/yeliu918/DHR) |
| [The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes.](https://arxiv.org/pdf/2012.14210) | Nils Reimers et al. | ACL 2021 |  NA |
| [Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection.](https://arxiv.org/pdf/2109.10739) | Negar Arabzadeh et al. | CIKM 2021 | [Python](https://github.com/Narabzad/Retrieval-Strategy-Selection) |
| [Boosted Dense Retriever.](https://arxiv.org/pdf/2112.07771.pdf) | Patrick Lewis et al. | Arxiv 2021 | NA |
| [PARM: A Paragraph Aggregation Retrieval Model for Dense Document-to-Document Retrieval.](https://arxiv.org/pdf/2201.01614) | Sophia Althammer et al. | ECIR 2022 | [Python](https://github.com/sophiaalthammer/parm) |
| [Sparsifying Sparse Representations for Passage Retrieval by Top-k Masking.](https://arxiv.org/pdf/2112.09628.pdf) | Jheng-Hong Yang et al. | Arxiv 2021 | NA |
| [Improving Document Representations by Generating Pseudo Query Embeddings for Dense Retrieval.](https://arxiv.org/pdf/2105.03599.pdf) | Hongyin Tang et al. | ACL 2021 | NA
| [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.](https://arxiv.org/pdf/2112.01488.pdf) | Keshav Santhanam et al. | Arxiv 2021 | [Python](https://github.com/stanford-futuredata/ColBERT)
| [GNN-encoder: Learning a Dual-encoder Architecture via Graph Neural Networks for Dense Passage Retrieval.](https://arxiv.org/pdf/2204.08241.pdf) | Jiduan Liu et al. | Arxiv 2022 | NA
| [Sentence-aware Contrastive Learning for Open-Domain Passage Retrieval.](https://aclanthology.org/2022.acl-long.76.pdf) | Bohong Wu et al. | ACL 2022 | [Python](https://github.com/chengzhipanpan/DCSR)
| [Aggretriever: A Simple Approach to Aggregate Textual Representation for Robust Dense Passage Retrieval.](https://arxiv.org/pdf/2208.00511.pdf) | Sheng-Chieh Lin et al. | Arxiv 2022 | [Python](https://github.com/castorini/dhr)
| [DPTDR: Deep Prompt Tuning for Dense Passage Retrieval.](https://arxiv.org/pdf/2208.11503.pdf) | Zhengyang Tang et al. | Arxiv 2022 | [Python](https://github.com/tangzhy/DPTDR)
| [LED: Lexicon-Enlightened Dense Retriever for Large-Scale Retrieval.](https://arxiv.org/pdf/2208.13661.pdf) | Kai Zhang et al. | Arxiv 2022 | NA
| [Task-Aware Specialization for Efficient and Robust Dense Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2210.05156.pdf) | Hao Cheng et al. | Arxiv 2022 | NA
| [COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List.](https://arxiv.org/pdf/2104.07186.pdf) | Luyu Gao et al. | NAACL 2021 | [Python](https://github.com/luyug/COIL) |
| [A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques.](https://arxiv.org/pdf/2106.14807.pdf) | Jimmy Lin et al. | Arxiv 2021 | NA
| [Pseudo Relevance Feedback with Deep Language Models and Dense Retrievers: Successes and Pitfalls.](https://arxiv.org/pdf/2108.11044.pdf) | Hang Li et al. | Arxiv 2021 | NA
| [Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback.](https://arxiv.org/pdf/2108.13454.pdf) | HongChien Yu et al. | CIKM 2021 | [Python](https://github.com/yuhongqian/ANCE-PRF)
| [Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval.](https://arxiv.org/pdf/2106.11251.pdf) | Xiao Wang et al. | SIGIR 2021 | NA
| [Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback: A Reproducibility Study.](https://arxiv.org/pdf/2112.06400.pdf) | Hang Li et al. | Arxiv 2021 | NA
| [Implicit Feedback for Dense Passage Retrieval: A Counterfactual Approach.](https://arxiv.org/pdf/2204.00718.pdf) | Shengyao Zhuang et al. | Arxiv 2022 | [Python](https://github.com/ielab/Counterfactual-DR)
| [Parameter-Efficient Prompt Tuning Makes Generalized and Calibrated Neural Text Retrievers.](https://arxiv.org/pdf/2207.07087.pdf) | Weng Lam Tam et al. | Arxiv 2022 | [Python](https://github.com/THUDM/P-tuning-v2/tree/main/PT-Retrieval)
| [Densifying Sparse Representations for Passage Retrieval by Representational Slicing.](https://arxiv.org/pdf/2112.04666.pdf) | Sheng-Chieh Lin et al. | Arxiv 2021 | NA
| [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.](https://arxiv.org/pdf/2107.05720) | Thibault Formal et al. | SIGIR 2021 | [Python](https://github.com/naver/splade) |
| [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval.](https://arxiv.org/pdf/2109.10086.pdf) | Thibault Formal et al. | Arxiv 2021 | [Python](https://github.com/naver/splade) |
| [BERT-based Dense Retrievers Require Interpolation with BM25 for Effective Passage Retrieval.](https://arvinzhuang.github.io/files/shuai2021interpolateDR.pdf) | Shuai Wang et al. | ICTIR 2021 | [Python](https://github.com/ielab/InterpolateDR-ICTIR2021) |
| [A White Box Analysis of ColBERT.](https://arxiv.org/pdf/2012.09650) | Thibault Formal et al. | ECIR 2021 | NA |
| [Towards Axiomatic Explanations for Neural Ranking Models.](https://arxiv.org/pdf/2106.08019) | Michael Völske et al. | ICTIR 2021 | [Python](https://github.com/webis-de/ICTIR-21) |
| [ABNIRML: Analyzing the Behavior of Neural IR Models.](https://arxiv.org/pdf/2011.00696.pdf) | Sean MacAvaney et al. | Arxiv 2020 | [Python](https://github.com/allenai/abnriml) |


## Training
### Formulation
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [More Robust Dense Retrieval with Contrastive Dual Learning.](https://arxiv.org/pdf/2107.07773.pdf) | Yizhi Li et al. | ICTIR 2021 | [Python](https://github.com/thunlp/DANCE) |
| [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval.](https://aclanthology.org/2021.findings-acl.191.pdf) | Ruiyang Ren et al. | ACL 2021 | [Python](https://github.com/PaddlePaddle/RocketQA/tree/main/research/PAIR_ACL2021) |
| [xMoCo: Cross Momentum Contrastive Learning for Open-Domain Question Answering.](https://aclanthology.org/2021.acl-long.477.pdf) | Nan Yang et al. |  ACL 2021 | NA |
| [A Modern Perspective on Query Likelihood with Deep Generative Retrieval Models.](https://arxiv.org/pdf/2106.13618) | Oleg Lesota et al. | ICTIR 2021 | [Python](https://github.com/CPJKU/DeepGenIR) |
| [Learning Diverse Document Representations with Deep Query Interactions for Dense Retrieval.](https://arxiv.org/pdf/2208.04232.pdf) | Zehan Li et al. | Arxiv 2022 | [Python](https://github.com/jordane95/dual-cross-encoder)
| [Shallow pooling for sparse labels.](https://arxiv.org/pdf/2109.00062.pdf) | Negar Arabzadeh et al. | Arxiv 2021 | NA
| [Hard Negatives or False Negatives: Correcting Pooling Bias in Training Neural Ranking Models.](https://dl.acm.org/doi/pdf/10.1145/3511808.3557343) | Yinqiong Cai et al. | Arxiv 2022 | NA
| [Debiased Contrastive Learning of Unsupervised Sentence Representations.](https://arxiv.org/pdf/2205.00656.pdf) | Kun Zhou et al. | ACL 2022 | NA


### Negative Selection
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently.](https://arxiv.org/pdf/2010.10469.pdf) | Jingtao Zhan et al. | Arxiv 2020 | NA |
| [Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2004.04906.pdf) | Vladimir Karpukhin et al. | EMNLP 2020 | [Python](https://github.com/facebookresearch/DPR) |
| [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval.](https://arxiv.org/pdf/2006.15498.pdf) | Jingtao Zhan et al. | Arxiv 2020 | [Python](https://github.com/jingtaozhan/RepBERT-Index) |
| [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval.](https://arxiv.org/pdf/2007.00808.pdf) | Lee Xiong et al. | ICLR 2021 | [Python](https://github.com/microsoft/ANCE) |
| [Optimizing Dense Retrieval Model Training with Hard Negatives.](https://arxiv.org/abs/2104.08051) | Jingtao Zhan et al. | SIGIR 2021 | [Python](https://github.com/jingtaozhan/DRhard) |
| [Neural Passage Retrieval with Improved Negative Contrast.](https://arxiv.org/pdf/2010.12523.pdf) | Jing Lu et al. | Arxiv 2020 | NA |
| [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2010.08191.pdf) | Yingqi Qu et al. | NAACL 2021 | [Python](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQA_NAACL2021) |
| [Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling.](https://arxiv.org/pdf/2104.06967.pdf) | Sebastian Hofstätter et al. | SIGIR 2021 | [Python](https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval) |
| [Scaling deep contrastive learning batch size under memory limited setup.](https://arxiv.org/pdf/2101.06983v2) | Luyu Gao et al. | RepL4NLP 2021 | [Python](https://github.com/luyug/GradCache) |
| [Multi-stage training with improved negative contrast for neural passage retrieval.](https://aclanthology.org/2021.emnlp-main.492.pdf) | Jing Lu et al. | EMNLP 2021 | NA |
| [Learning robust dense retrieval models from incomplete relevance labels.](https://dl.acm.org/doi/pdf/10.1145/3404835.3463106) | Prafull Prakash et al. | SIGIR 2021 | [Python](https://github.com/purble/RANCE) |
| [Efficient Training of Retrieval Models Using Negative Cache.](https://papers.nips.cc/paper/2021/file/2175f8c5cd9604f6b1e576b252d4c86e-Paper.pdf) | Erik M. Lindgren et al. | NeurIPS 2021 | [Python](NA) |
| [CODER: An efficient framework for improving retrieval through COntextual Document Embedding Reranking.](https://arxiv.org/pdf/2112.08766) | George Zerveas et al. | Arxiv 2021 | NA |
| [Curriculum Learning for Dense Retrieval Distillation.](https://arxiv.org/pdf/2204.13679.pdf) | Hansi Zeng et al. | SIGIR 2022 | [Python](https://github.com/HansiZeng/CL-DRD)
| [SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval.](https://arxiv.org/pdf/2210.11773.pdf) | Kun Zhou et al. | EMNLP 2022 | [Python](https://github.com/microsoft/SimXNS) |

### Data Augmentation
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [UniK-QA: Unified Representations of Structured and Unstructured Knowledge for Open-Domain Question Answering.](https://arxiv.org/pdf/2012.14610.pdf) | Barlas Oguz et al. | Arxiv 2021 | NA |
| [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks.](https://arxiv.org/pdf/2010.08240.pdf) | Nandan Thakur et al. | NAACL 2021 | [Python](www.sbert.net) |
| [Is Retriever Merely an Approximator of Reader?](https://arxiv.org/pdf/2010.10999.pdf) | Sohee Yang et al. | Arxiv 2020 | NA |
| [Distilling Knowledge from Reader to Retriever for Question Answering.](https://openreview.net/pdf?id=NTEz-6wysdb) | Gautier Izacard et al. | ICLR 2021 | [Python](github.com/facebookresearch/FiD) |
| [Distilling Knowledge for Fast Retrieval-based Chat-bots.](https://arxiv.org/pdf/2004.11045.pdf) | Amir Vakili Tahami et al. | SIGIR 2020 | [Python](https://github.com/KamyarGhajar/DistilledNeuralResponseRanker) |
| [Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation.](https://arxiv.org/pdf/2010.02666.pdf) | Sebastian Hofstätter et al. | Arxiv 2020 | [Python](https://github.com/sebastian-hofstaetter/neural-ranking-kd) |
| [Distilling Dense Representations for Ranking using Tightly-Coupled Teachers.](https://arxiv.org/pdf/2010.11386.pdf) | Sheng-Chieh Lin et al. | Arxiv 2020 | [Python](https://github.com/castorini/pyserini/blob/master/docs/experiments-tctcolbert) |
| [In-Batch Negatives for Knowledge Distillation with Tightly-Coupled Teachers for Dense Retrieval.](https://aclanthology.org/2021.repl4nlp-1.17.pdf) | Sheng-Chieh Lin et al. | RepL4NLP 2021 | [Python](https://github.com/castorini/pyserini/blob/master/docs/experiments-tct_colbert-v2.md) |
| [Neural Retrieval for Question Answering with Cross-Attention Supervised Data Augmentation.](https://arxiv.org/pdf/2009.13815.pdf) | Yinfei Yang et al. | ACL 2021 | NA |
| [Enhancing Dual-Encoders with Question and Answer Cross-Embeddings for Answer Retrieval.](https://aclanthology.org/2021.findings-emnlp.198.pdf) | Yanmeng Wang et al. | EMNLP 2021 | NA |
| [Pseudo Label based Contrastive Sampling for Long Text Retrieval.](https://ieeexplore.ieee.org/abstract/document/9675219) | Le Zhu et al. | IALP 2021 | NA |
| [Multi-View Document Representation Learning for Open-Domain Dense Retrieval.](https://arxiv.org/pdf/2203.08372.pdf) | Shunyu Zhang et al. | ACL 2022 | NA |
| [Augmenting Document Representations for Dense Retrieval with Interpolation and Perturbation.](https://arxiv.org/pdf/2203.07735v2.pdf) | Soyeong Jeong et al. | ACL 2022 | [Python](github.com/starsuzi/DAR) |
| [ERNIE-Search: Bridging Cross-Encoder with Dual-Encoder via Self On-the-fly Distillation for Dense Passage Retrieval.](https://arxiv.org/pdf/2205.09153.pdf) | Yuxiang Lu et al. | Arxiv 2022 | NA |
| [Pro-KD: Progressive Distillation by Following the Footsteps of the Teacher.](https://arxiv.org/pdf/2110.08532.pdf) | Mehdi Rezagholizadeh et al. | COLING 2022 | NA
| [Questions Are All You Need to Train a Dense Passage Retriever.](https://arxiv.org/pdf/2206.10658.pdf) | Devendra Singh Sachan et al. | Arxiv 2022 | [Python](https://github.com/DevSinghSachan/art)
| [PROD: Progressive Distillation for Dense Retrieval.](https://arxiv.org/pdf/2209.13335.pdf) | Zhenghao Lin et al. | Arxiv 2022 | NA
| [Answering Open-Domain Questions of Varying Reasoning Steps from Text.](https://arxiv.org/pdf/2010.12527.pdf) | Peng Qi et al. | EMNLP 2021 | [Python](https://github.com/beerqa/IRRR) |
| [Multi-Task Retrieval for Knowledge-Intensive Tasks.](https://arxiv.org/pdf/2101.00117.pdf) | Jean Maillard et al. | ACL 2021 | NA

### Pre-training
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) | Kenton Lee et al. | ACL 2019 | [Python](https://github.com/google-research/language/blob/master/language/orqa/README.md) |
| [Pre-training tasks for embedding-based large scale retrieval.](https://arxiv.org/pdf/2002.03932.pdf) | Wei-Cheng Chang et al. | ICLR 2020 | NA |
| [PROP: Pre-training with Representative Words Prediction for Ad-hoc Retrieval.](https://arxiv.org/pdf/2010.10137.pdf) | Xinyu Ma et al. | WSDM 2021 | [Python](https://github.com/Albert-Ma/PROP) |
| [B-PROP: Bootstrapped Pre-training with Representative Words Prediction for Ad-hoc Retrieval.](https://arxiv.org/pdf/2104.09791.pdf) | Xinyu Ma et al. | SIGIR 2021 | NA |
| [Domain-matched Pre-training Tasks for Dense Retrieval.](https://arxiv.org/pdf/2107.13602.pdf) | Barlas Oguz et al. | Arxiv 2021 | NA |
| [Less is More: Pre-train a Strong Text Encoder for Dense Retrieval Using a Weak Decoder.](https://arxiv.org/pdf/2102.09206.pdf) | Shuqi Lu et al. | EMNLP 2021 | [Python](https://github.com/microsoft/SEED-Encoder/) |
| [Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models.](https://arxiv.org/pdf/2108.08877) | Jianmo Ni et al. | Arxiv 2021 | [Python](https://github.com/google-research/text-to-text-transfer-transformer) |
| [Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval.](https://arxiv.org/pdf/2108.05540.pdf) | Luyu Gao et al. | ACL 2022 | [Python](https://github.com/luyug/Condenser) |
| [Condenser: a Pre-training Architecture for Dense Retrieval.](https://arxiv.org/pdf/2104.08253.pdf) | Luyu Gao et al. | EMNLP 2021 | [Python](https://github.com/luyug/Condenser) |
| [TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning.](https://arxiv.org/pdf/2104.06979) | Kexin Wang et al. | EMNLP 2021 | [Python](https://github.com/UKPLab/sentence-transformers/) |
| [SimCSE: Simple Contrastive Learning of Sentence Embeddings.](https://arxiv.org/pdf/2104.08821.pdf) | Tianyu Gao et al. | EMNLP 2021 | [Python](https://github.com/princeton-nlp/SimCSE) |
| [Towards Robust Neural Retrieval Models with Synthetic Pre-Training.](https://arxiv.org/pdf/2104.07800v1) | Revanth Gangi Reddy et al. | Arxiv 2021 | NA |
| [Hyperlink-induced Pre-training for Passage Retrieval in Open-domain Question Answering.](https://arxiv.org/pdf/2203.06942.pdf) | Jiawei Zhou et al. | ACL 2022 | [Python](https://github.com/jzhoubu/HLP) |
| [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.](https://arxiv.org/pdf/1908.10084.pdf) | Nils Reimer et al. | EMNLP 2019 | [Python](https://github.com/UKPLab/sentence-transformers)
| [Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions using Enhanced Reduction.](https://arxiv.org/pdf/2203.13088.pdf) | Sebastian Hofstätter et al. | Arxiv 2022 | [Python](https://github.com/sebastian-hofstaetter/colberter)
| [Learning to Retrieve Passages without Supervision.](https://arxiv.org/pdf/2112.07708.pdf) | Ori Ram et al. | Arxiv 2021 |  [Python](https://github.com/oriram/spider)
| [Text and Code Embeddings by Contrastive Pre-Training.](https://arxiv.org/pdf/2201.10005.pdf) | Arvind Neelakantan et al. | Arxiv 2022 | NA
| [Pre-train a Discriminative Text Encoder for Dense Retrieval via Contrastive Span Prediction.](https://arxiv.org/pdf/2204.10641.pdf) | Xinyu Ma et al. | SIGIR 2022 | [Python](https://github.com/Albert-Ma/COSTA)
| [RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder.](https://arxiv.org/pdf/2205.12035.pdf) | Shitao Xiao et al. | CoRR 2022 | 
| [SIMLM: Pre-training with Representation Bottleneck for Dense Passage Retrieval.](https://arxiv.org/pdf/2207.02578.pdf) | Liang Wang et al. | CoRR 2022 | [Python](https://github.com/microsoft/unilm)
| [Masked Autoencoders As The Unified Learners For Pre-Trained Sentence Representation.](https://arxiv.org/pdf/2208.00231.pdf) | Alexander Liu et al. | CoRR 2022 | NA
| [LEXMAE: Lexicon-BottleNecked Pretraining fot Large-scale Retrieval.](https://arxiv.org/pdf/2208.14754.pdf) | Tao Shen et al. | Arxiv 2022 | NA
| [PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them.](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00415/107615/PAQ-65-Million-Probably-Asked-Questions-and-What) | Patrick Lewis et al. | Arxiv 2021 | [Python](https://github.com/facebookresearch/PAQ)
| [End-to-End Synthetic Data Generation for Domain Adaptation of Question Answering Systems.](https://arxiv.org/pdf/2010.06028.pdf) | Siamak Shakeri et al. | EMNLP 2020 | NA
| [ConTextual Mask Auto-Encoder for Dense Passage Retrieval.](https://arxiv.org/pdf/2208.07670.pdf) | Xing Wu et al. | Arxiv 2022 | [Python](https://github.com/caskcsg/ir)
| [A Contrastive Pre-training Approach to Learn Discriminative Autoencoder for Dense Retrieval.](https://arxiv.org/pdf/2208.09846.pdf) | Xinyu Ma et al. | Arxiv 2022 | NA


## Indexing
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Learning Passage Impacts for Inverted Indexes.](https://arxiv.org/pdf/2104.12016.pdf) | Antonio Mallia et al. | SIGIR 2021 | [Python](https://github.com/DI4IR/SIGIR2021) |
| [Accelerating Large-Scale Inference with Anisotropic Vector Quantization.](https://arxiv.org/pdf/1908.10396) | Ruiqi Guo et al. | Arxiv 2019 | [Python](https://github.com/google-research/google-research/tree/master/scann) |
| [Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance.](https://arxiv.org/pdf/2108.00644.pdf) | Jingtao Zhan et al. | CIKM 2021 | [Python](https://github.com/jingtaozhan/JPQ) |
| [Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3488560.3498443) | Jingtao Zhan et al. | WSDM 2022 | [Python](https://github.com/jingtaozhan/repconc) |
| [Joint Learning of Deep Retrieval Model and Product Quantization based Embedding Index.](https://arxiv.org/pdf/2105.03933) | Han Zhang et al. | SIGIR 2021 | [Python](https://github.com/jdcomsearch/poeem) |
| [Efficient Passage Retrieval with Hashing for Open-domain Question Answering.](https://arxiv.org/pdf/2106.00882) | Ikuya Yamada et al. | ACL 2021 | [Python](https://github.com/studio-ousia/bpr) |
| [A Memory Efficient Baseline for Open Domain Question Answering.](https://arxiv.org/pdf/2012.15156.pdf) | Gautier Izacard et al. | Arxiv 2020 | NA |
| [Simple and Effective Unsupervised Redundancy Elimination to Compress Dense Vectors for Passage Retrieval.](https://aclanthology.org/2021.emnlp-main.227.pdf) | Xueguang Ma et al. | EMNLP 2021 | [Python](http://pyserini.io/) |
| [The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes.](https://arxiv.org/pdf/2012.14210.pdf) | Nils Reimers et al. | ACL 2021 | NA |
| [Matching-oriented Product Quantization For Ad-hoc Retrieval.](https://arxiv.org/pdf/2104.07858) | Shitao Xiao et al. | EMNLP 2021 | [Python](https://github.com/microsoft/MoPQ) |
| [Progressively Optimized Bi-Granular Document Representation for Scalable Embedding Based Retrieval.](https://arxiv.org/pdf/2201.05409v1) | Shitao Xiao et al. | WWW 2022 | NA |
| [Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS).](https://proceedings.neurips.cc/paper/2014/file/310ce61c90f3a46e340ee8257bc70e93-Paper.pdf) | Anshumali Shrivastava et al. | NeuraIPS 2014 | NA
| [ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.](https://arxiv.org/pdf/1807.05614.pdf) | Martin Aumüller et al. | SISAP 2017 | NA
| [Results of the NeurIPS’21 Challenge on Billion-Scale Approximate Nearest Neighbor Search.](https://arxiv.org/pdf/2205.03763.pdf) | Harsha Vardhan Simhadri et al. | Arxiv 2022 | [Python](https://github.com/harsha-simhadri/big-ann-benchmarks/)
| [Interpreting Dense Retrieval as Mixture of Topics.](https://arxiv.org/pdf/2111.13957.pdf) | Jingtao Zhan et al. | Arxiv 2021 | NA
| [The Web Is Your Oyster - Knowledge-Intensive NLP against a Very Large Web Corpus.](https://arxiv.org/pdf/2112.09924.pdf) | Aleksandra Piktus et al. | CoRR 2021 | NA
| [Bi-Phase Enhanced IVFPQ for Time-Efficient Ad-hoc Retrieval.](https://arxiv.org/pdf/2210.05521.pdf) | Peitian Zhang et al. | Arxiv 2022 | NA

## Interation with Re-ranking
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking.](https://aclanthology.org/2021.emnlp-main.224.pdf) | Ruiyang Ren et al. | EMNLP 2021 | [Python](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQAv2_EMNLP2021) |
| [Dealing with Typos for BERT-based Passage Retrieval and Ranking.](https://arxiv.org/pdf/2108.12139.pdf) | Shengyao Zhuang et al. | EMNLP 2021 | [Python](https://github.com/ielab/typos-aware-BERT) |
| [Trans-Encoder: Unsupervised sentence-pair modelling through self- and mutual-distillations.](https://arxiv.org/pdf/2109.13059.pdf) | Fangyu Liu et al. | ICLR 2022 | [Python](https://github.com/amzn/trans-encoder) |
| [Adversarial Retriever-Ranker for dense text retrieval.](https://arxiv.org/pdf/2110.03611.pdf) | Hang Zhang et al. | Arxiv 2021 | NA |
| [Embedding-based Retrieval in Facebook Search.](https://dl.acm.org/doi/pdf/10.1145/3394486.3403305) | Jui-Ting Huang et al. | KDD 2020 | NA |
| [Passage Re-ranking With BERT.](https://arxiv.org/pdf/1901.04085.pdf) | Rodrigo Nogueira et al. | Arxiv 2019 | NA
| [Understanding the Behaviors of BERT in Ranking.](https://arxiv.org/pdf/1904.07531) | Yifan Qiao et al. | CoRR 2019 | NA
| [Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering.](https://arxiv.org/pdf/1908.08167.pdf) | Zhiguo Wang et al. | Arxiv 2019 | NA
| [TOWARDS ROBUST RANKER FOR TEXT RETRIEVAL.](https://arxiv.org/pdf/2206.08063.pdf) | Yucheng Zhou et al. | Arxiv 2022 | [Python](https://github.com/taoshen58/R2ANKER)
| [Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline.](https://arxiv.org/pdf/2101.08751.pdf) | Luyu Gao et al. | ECIR 2021 | NA
| [Multi-Stage Document Ranking with BERT.](https://arxiv.org/pdf/1910.14424.pdf) | Rodrigo Nogueira et al. | CoRR 2019 | NA


## Advanced Topics

### Zero-shot Dense Retrieval
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models.](https://arxiv.org/pdf/2104.08663.pdf) | Nandan Thakur et al. | NIPS 2021 | [Python](https://github.com/UKPLab/beir) |
| [A Thorough Examination on Zero-shot Dense Retrieval.](https://arxiv.org/pdf/2204.12755.pdf) | Ruiyang Ren et al. | Arxiv 2022 | NA |
| [Challenges in Generalization in Open Domain Question Answering.](https://arxiv.org/pdf/2109.01156.pdf) | Linqing Liu et al. | NAACL 2022 | [Python](https://github.com/likicode/QA-generalize) |
| [Zero-shot Neural Passage Retrieval via Domain-targeted Synthetic Question Generation.](https://arxiv.org/pdf/2004.14503.pdf) | Ji Ma et al. | Arxiv 2021 | NA |
| [Efficient Retrieval Optimized Multi-task Learning.](https://arxiv.org/pdf/2104.10129) | Hengxin Fun et al. | Arxiv 2021 | NA |
| [Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations.](https://arxiv.org/pdf/2110.07581.pdf) | Ji Xin et al. | ACL 2022 | NA |
| [Towards Robust Neural Retrieval Models with Synthetic Pre-Training.](https://arxiv.org/pdf/2104.07800v1.pdf) | Revanth Gangi Reddy et al. | Arxiv 2021 | NA |
| [Embedding-based Zero-shot Retrieval through Query Generation.](https://arxiv.org/pdf/2009.10270.pdf) | Davis Liang et al. | Arxiv 2020 | NA |
| [GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval.](https://arxiv.org/pdf/2112.07577.pdf) | Kexin Wang et al. | Arxiv 2021 | [Python](https://github.com/UKPLab/gpl) |
| [Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?](https://arxiv.org/pdf/2110.06918.pdf) | Xilun Chen et al. | Arxiv 2021 | [Python](https://github.com/facebookresearch/dpr-scale/tree/main/spar) |
| [LaPraDoR: Unsupervised Pretrained Dense Retriever for Zero-Shot Text Retrieval.](https://arxiv.org/pdf/2203.06169v2.pdf) | Canwen Xu et al. | ACL 2022 | [Python](https://github.com/JetRunner/LaPraDoR) |
| [Out-of-Domain Semantics to the Rescue! Zero-Shot Hybrid Retrieval Models.](https://arxiv.org/pdf/2201.10582.pdf) | Tao Chen et al. | ECIR 2022 | NA |
| [Towards Unsupervised Dense Information Retrieval with Contrastive Learning.](https://arxiv.org/pdf/2112.09118v1.pdf) | Gautier Izacard et al. | Arxiv 2021 | NA |
| [Large Dual Encoders Are Generalizable Retrievers.](https://arxiv.org/pdf/2112.07899.pdf) | Jianmo Ni et al. | Arxiv 2021 | NA |
| [KILT: a Benchmark for Knowledge Intensive Language Tasks.](https://arxiv.org/pdf/2009.02252.pdf) | Fabio Petroni et al. | Arxiv 2020 | [Python](https://github.com/facebookresearch/KILT)
| [Promptagator: Few-shot Dense Retrieval From 8 Examples.](https://arxiv.org/pdf/2209.11755.pdf) | Zhuyun Dai et al. | Arxiv 2022 | NA

### Improving the Robustness to Query Variations
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Towards Robust Dense Retrieval via Local Ranking Alignment.](https://www.ijcai.org/proceedings/2022/0275.pdf) | Xuanang Chen et al. | IJCAI 2022 | [Python](https://github.com/cxa-unique/RoDR)
| [CharacterBERT and Self-Teaching for Improving the Robustness of Dense Retrievers on Queries with Typos.](https://arxiv.org/pdf/2204.00716.pdf) | Shengyao Zhuang et al. | SIGIR 2022 | [Python](https://github.com/ielab/CharacterBERT-DR)
| [Evaluating the Robustness of Retrieval Pipelines with Query Variation Generators.](https://arxiv.org/pdf/2111.13057) | Gustavo Penha et al. | ECIR 2022 | [Python](https://github.com/Guzpenha/query_variation_generators)
| [Retrieval Consistency in the Presence of Query Variations.](https://people.eng.unimelb.edu.au/ammoffat/abstracts/bmst17sigir.pdf) | Peter Bailey et al. | SIGIR 2017 | NA
| [Analysing the Robustness of Dual Encoders for Dense Retrieval Against Misspellings.](https://arxiv.org/pdf/2205.02303.pdf) | Peter Bailey et al. | Arxiv 2022 | [Shell](https://github.com/GSidiropoulos/dense-retrieval-against-misspellings)
| [A Survey of Automatic Query Expansion in Information Retrieval.](http://www-labs.iro.umontreal.ca/~nie/IFT6255/carpineto-Survey-QE.pdf) | Claudio Carpineto et al. | CSUR 2012 | NA
| [BERT Rankers are Brittle: a Study using Adversarial Document Perturbations.](https://arxiv.org/pdf/2206.11724.pdf) | Yumeng Wang et al. | SIGIR 2022 | [Python](https://github.com/menauwy/brittlebert)
| [Order-Disorder: Imitation Adversarial Attacks for Black-box Neural Ranking Models.](https://arxiv.org/pdf/2209.06506.pdf) | Jiawei Liu et al. | CoRR 2022 | NA

### Generative Text Retrieval
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Transformer Memory as a Diﬀerentiable Search Index.](https://arxiv.org/pdf/2202.06991.pdf) | Yi Tay et al. | Arxiv 2022 | NA |
| [DynamicRetriever: A Pre-training Model-based IR System with Neither Sparse nor Dense Index.](https://arxiv.org/pdf/2203.00537) | Yujia Zhou et al. | Arxiv 2022 | NA |
| [Autoregressive Search Engines: Generating Substrings as Document Identifiers.](https://arxiv.org/pdf/2204.10628.pdf) | Michele Bevilacqua et al. | Arxiv 2022 | [Python](https://github.com/facebookresearch/SEAL) |
| [Generative Retrieval for Long Sequences.](https://arxiv.org/pdf/2204.13596.pdf) | Hyunji Lee et al. | Arxiv 2022 | NA |
| [GERE: Generative Evidence Retrieval for Fact Verification.](https://arxiv.org/pdf/2204.05511.pdf) | Jiangui Chen et al. | SIGIR 2022 | [Python](https://github.com/Chriskuei/GERE) |
| [Autoregressive Entity Retrieval.](https://arxiv.org/abs/2010.00904) | Nicola De Cao et al. | ICLR 2021 | [Python](https://github.com/facebookresearch/GENRE) |
| [Rethinking Search: Making Domain Experts out of Dilettantes.](https://arxiv.org/pdf/2105.02274.pdf) | Donald Metzler et al. | SIGIR 2021 | NA
| [Transformer Memory as a Differentiable Search Index.](https://arxiv.org/pdf/2202.06991.pdf) | Yi Tay et al. | Arxiv 2022 | NA
| [A Neural Corpus Indexer for Document Retrieval.](https://arxiv.org/pdf/2206.02743.pdf) | Yujing Wang et al. | CoRR 2022 | NA
| [Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation.](https://arxiv.org/pdf/2206.10128.pdf) | Shengyao Zhuang et al. | CoRR 2022 | [Python](https://github.com/ArvinZhuang/DSI-QG)
| [Ultron: An Ultimate Retriever on Corpus with a Model-based Indexer.](https://arxiv.org/pdf/2208.09257.pdf) | Yujia Zhou et al. | CoRR 2022 | NA
| [CorpusBrain: Pre-train a Generative Retrieval Model for Knowledge-Intensive Language Tasks.](https://dl.acm.org/doi/pdf/10.1145/3511808.3557271) | Jiangui Chen et al. | Arxiv 2022 | [Python](https://github.com/ict-bigdatalab/CorpusBrain)

### Retrieval-Augmented Language Model
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Generalization through memorization: Nearest neighbor language models.](https://arxiv.org/pdf/1911.00172) | Urvashi Khandelwa et al. | Arxiv 2020 | [Python](https://github.com/urvashik/knnlm) |
| [Adaptive semiparametric language models.](https://aclanthology.org/2021.tacl-1.22.pdf) | Dani Yogatama et al. | TACL 2021 | NA |
| [Improving language models by retrieving from trillions of tokens.](https://arxiv.org/pdf/2112.04426.pdf) | Borgeaud, Sebastian, et al. | Arxiv 2021 | NA |
| [REALM: Retrieval-Augmented Language Model Pre-Training.](https://arxiv.org/pdf/2002.08909.pdf) | Kelvin Guu et al. | ICML 2020 | [Python](https://github.com/google-research/language/blob/master/language/realm/README.md) |
| [Simple and Efficient ways to Improve REALM.](https://arxiv.org/abs/2104.08710.pdf) | Vidhisha Balachandran et al. | Arxiv 2021 | NA |
| [Adaptive Semiparametric Language Models.](https://dyogatama.github.io/publications_files/Yogatama+etal_TACL2021.pdf) | Dani Yogatama et al. | TACL 2021 | NA
| [Efficient Nearest Neighbor Language Models.](https://arxiv.org/pdf/2109.04212) | Junxian He et al. | EMNLP 2021 | [Python](https://github.com/jxhe/efficient-knnlm)



## Applications
### Information Retrieval Applications
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models.](https://arxiv.org/pdf/2108.04049v1.pdf) | Bogdan Kostic et al. | Arxiv 2021 | NA |
| [Open Domain Question Answering over Tables via Dense Retrieval.](https://aclanthology.org/2021.naacl-main.43.pdf) | Jonathan Herzig et al. | NAACL 2021 | [Python](https://github.com/google-research/tapas) |
| [SituatedQA: Incorporating Extra-Linguistic Contexts into QA.](https://arxiv.org/pdf/2109.06157.pdf) | Michael J.Q. Zhang et al. | EMNLP 2021 | [DATA](https://situatedqa.github.io/) |
| [XOR QA: Cross-lingual Open-Retrieval Question Answering.](https://aclanthology.org/2021.naacl-main.46.pdf) | Akari Asai et al. | NAACL 2021 | [Python](https://nlp.cs.washington.edu/xorqa/) |
| [One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval.](https://arxiv.org/pdf/2107.11976.pdf) | Akari Asai et al. | NeurIPS 2021 | [Python](https://github.com/AkariAsai/CORA) |
| [Evaluating Token-Level and Passage-Level Dense Retrieval Models for Math Information Retrieval.](https://arxiv.org/pdf/2203.11163.pdf) | Wei Zhong et al. | Arxiv 2022 | [Python](https://github.com/approach0/math-dense-retrievers) |
| [ReACC: A Retrieval-Augmented Code Completion Framework.](https://arxiv.org/pdf/2203.07722.pdf) | Shuai Lu et al. | ACL 2022 | [Python](https://github.com/microsoft/ReACC)
| [Improving Biomedical Information Retrieval with Neural Retrievers.](https://arxiv.org/pdf/2201.07745.pdf) | Man Luo et al. | AAAI 2022 | NA |
| [A Comprehensive Survey on Cross-modal Retrieval.](https://arxiv.org/pdf/1607.06215) | Kaiye Wang et al. | CoRR 2016 | NA

### Natural Language Processing Applications
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) | Patrick Lewis et al. | Arxiv 2020 | NA
| [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering.](https://aclanthology.org/2021.eacl-main.74.pdf) | Gautier Izacard et al. | ECAL 2021 | [Python](https://github.com/facebookresearch/FiD) |
| [End-to-End Training of Neural Retrievers for Open-Domain Question Answering.](https://arxiv.org/pdf/2101.00408.pdf) | Devendra Singh Sachan et al. | ACL 2021 | [Python](https://github.com/NVIDIA/Megatron-LM)
| [Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval.](https://arxiv.org/pdf/2101.00436.pdf) | Omar Khattab et al. | NeurIPS 2021 | [Python](https://github.com/stanford-futuredata/Baleen) |
| [Answering Complex Open-domain Questions with Multi-hop Dense Retrieval.](https://arxiv.org/pdf/2009.12756.pdf) | Wenhan Xiong et al. | ICLR 2021 | [Python](https://github.com/facebookresearch/multihop_dense_retrieval)
| [Learning Dense Representations for Entity Retrieval.](https://aclanthology.org/K19-1049.pdf) | Daniel Gillick et al. | CoNLL 2019 | NA |
| [Scalable Zero-shot Entity Linking with Dense Entity Retrieval.](https://arxiv.org/pdf/1911.03814.pdf) | Ledell Wu et al. | EMNLP 2020 | [Python](https://github.com/facebookresearch/BLINK) |
| [Zero-Shot Entity Linking by Reading Entity Descriptions.](https://arxiv.org/pdf/1906.07348.pdf) | Lajanugen Logeswaran et al. | ACL 2019 | [Python](https://github.com/lajanugen/zeshel) |
| [Retrieval Augmentation Reduces Hallucination in Conversation.](https://arxiv.org/pdf/2104.07567.pdf) | Kurt Shuster et al. | EMNLP 2021 | NA |
| [Internet-Augmented Dialogue Generation.](https://arxiv.org/pdf/2107.07566) | Mojtaba Komeili et al. | ACL 2022 | NA |
| [LaMDA: Language Models for Dialog Applications.](https://arxiv.org/pdf/2201.08239.pdf) | Romal Thoppilan et al. | Arxiv 2022 | NA |

### Industrial Practice
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Pre-trained Language Model for Web-scale Retrieval in Baidu Search.](https://arxiv.org/pdf/2106.03373v3) | Yiding Liu et al. | KDD 2021 | NA |
| [MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search.](http://research.baidu.com/Public/uploads/5d12eca098d40.pdf) | Miao Fan. | KDD 2019 | NA |
| [Uni-Retriever: Towards Learning The Unified Embedding Based Retriever in Bing Sponsored Search.](https://arxiv.org/pdf/2202.06212.pdf) | Jianjin Zhang et al. | Arxiv 2022 | NA |
| [Embedding-based Product Retrieval in Taobao Search.](https://arxiv.org/pdf/2106.09297.pdf) | Sen Li et al. | KDD 2021 | NA |
| [Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook.](https://scontent-nrt1-1.xx.fbcdn.net/v/t39.8562-6/246795273_2109661252514735_2459553109378891559_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=7LLAz1SvhvcAX9Dr2-E&_nc_ht=scontent-nrt1-1.xx&oh=00_AT_sJBUEVm6mlAYngNn31Oc2BTqokLB9dvcdHTLYsIDCqA&oe=629847E3) | Yiqun Liu et al. | KDD 2021 | NA |
| [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.](https://suhasjs.github.io/files/diskann_neurips19.pdf) | Suhas Jayaram Subramanya et al. | NeurIPS 2019 | [Python](https://github.com/Microsoft/DiskANN) |
| [SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search.](https://arxiv.org/pdf/2111.08566.pdf) | Qi Chen et al. | NeurIPS 2021 | [Python](https://github.com/microsoft/SPTAG) |
| [HEARTS: Multi-task Fusion of Dense Retrieval and Non-autoregressive Generation for Sponsored Search.](https://arxiv.org/pdf/2209.05861.pdf) | Bhargav Dodla et al. | Arxiv 2022 | NA
| [Sponsored Search Auctions: Recent Advances and Future Directions.](https://taoqin.github.io/papers/ssa-new.pdf) | Tao Qin et al. | TIST 2015 | NA
| [Semantic Retrieval at Walmart.](https://dl.acm.org/doi/abs/10.1145/3534678.3539164) | Alessandro Magnani et al. | KDD 2022 | NA


## Datasets
| **Paper** | **Author** | **Venue** | **Link** |
| --- | --- | --- | --- |
| [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models.](https://arxiv.org/pdf/2104.08663.pdf) | Nandan Thakur et al. | NeurIPS 2021 | [DATA](https://github.com/beir-cellar/beir) |
| [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.](https://arxiv.org/pdf/1611.09268.pdf) | Payal Bajaj et al. | NeurIPS 2016 | [DATA](https://microsoft.github.io/msmarco/) |
| [Natural Questions: a Benchmark for Question Answering Research.](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf) | Tom Kwiatkowski et al. | TACL 2019 | [DATA](https://ai.google.com/research/NaturalQuestions) |
| [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.](https://arxiv.org/pdf/1705.03551) | Mandar Joshi et al. | ACL 2017 | [DATA](http://nlp.cs.washington.edu/triviaqa/) |
| [mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset.](https://arxiv.org/pdf/2108.13897.pdf) | Luiz Henrique Bonifacio et al. | Arxiv 2021 | [DATA](https://github.com/unicamp-dl/mMARCO.git) |
| [TREC 2019 News Track Overview.](https://trec.nist.gov/pubs/trec28/papers/OVERVIEW.N.pdf) | Ian Soborof et al. | TREC 2019 | [DATA](https://trec.nist.gov/data/wapost/) |
| [TREC-COVID: rationale and structure of an information retrieval shared task for COVID-19.](https://academic.oup.com/jamia/article-pdf/27/9/1431/34153771/ocaa091.pdf) | Kirk Roberts et al. | J Am Med Inform Assoc. 2020 | [DATA](https://ir.nist.gov/covidSubmit/) |
| [A Full-Text Learning to Rank Dataset for Medical Information Retrieval.](https://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf) | Vera Boteva et al. | ECIR 2016 | [DATA](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) |
| [A Data Collection for Evaluating the Retrieval of Related Tweets to News Articles.](https://research.signal-ai.com/publications/signal1m-tweet-retrieval.pdf) | Axel Suarez et al. | ECIR 2018 | [DATA](https://github.com/igorbrigadir/newsir16-data/tree/master/) |
| [Overview of Touché 2020: Argument Retrieval.](http://ceur-ws.org/Vol-2696/paper_261.pdf) | Alexander Bondarenko et al. | CLEF 2020 | [DATA](https://touche.webis.de/) |
| [Retrieval of the Best Counterargument without Prior Topic Knowledge.](https://aclanthology.org/P18-1023.pdf) | Henning Wachsmuth et al. | ACL 2018 | [DATA](http://www.arguana.com/) |
| [DBpedia-Entity v2: A Test Collection for Entity Search.](https://dl.acm.org/doi/pdf/10.1145/3077136.3080751) | Faegheh Hasibi et al. | SIGIR 2017 | [DATA](https://iai-group.github.io/DBpedia-Entity/) |
| [ORCAS: 20 Million Clicked Query-Document Pairs for Analyzing Search.](https://dl.acm.org/doi/abs/10.1145/3340531.3412779) | Nick Craswell et al. | CIKM 2020 | [DATA](https://microsoft.github.io/TREC-2020-Deep-Learning/ORCAS) |
| [TREC 2022 Deep Learning Track Guidelines.](https://www.microsoft.com/en-us/research/uploads/prod/2022/05/trec2021-deeplearning-overview.pdf) | Nick Craswell et al. | TREC 2021 | [DATA](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2021) |
| [DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine.](https://arxiv.org/pdf/2203.10232) | Yifu Qiu et al. | Arxiv 2022 | [DATA](https://github.com/baidu/DuReader/tree/master/DuReader-Retrieval) |
| [SQuAD: 100,000+ Questions for Machine Comprehension of Text.](https://arxiv.org/pdf/1606.05250.pdf) | Pranav Rajpurkar et al. | EMNLP 2016 | [DATA](https://rajpurkar.github.io/SQuAD-explorer/) |
| [HOTPOTQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.](https://arxiv.org/pdf/1809.09600.pdf) | Zhilin Yang et al. | EMNLP 2018 | [DATA](https://hotpotqa.github.io/) |
| [Semantic Parsing on Freebase from Question-Answer Pairs.](https://aclanthology.org/D13-1160.pdf) | Jonathan Berant et al. | EMNLP 2013 | [DATA](http://nlp.stanford.edu/software/sempre/) |
| [Modeling of the Question Answering Task in the YodaQA System.](https://link.springer.com/content/pdf/10.1007/978-3-319-24027-5.pdf) | Petr Baudiš et al. | CLEF 2015 | [DATA](https://github.com/brmson/dataset-factoid-curated) |
| [WWW'18 Open Challenge: Financial Opinion Mining and Question Answering.](https://dl.acm.org/doi/pdf/10.1145/3184558.3192301) | Macedo Maia et al. | WWW 2018 | [DATA](https://sites.google.com/view/fiqa/home) |
| [An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition.](https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-015-0564-6.pdf) | George Tsatsaronis et al. | BMC Bioinform. 2015 | [DATA](http://participants-area.bioasq.org/) |
| [CQADupStack: A Benchmark Data Set for Community Question-Answering Research.](https://dl.acm.org/doi/pdf/10.1145/2838931.2838934) | Doris Hoogeveen et al. | ADCS 2015 | [DATA](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) |
| [First Quora Dataset Release: Question Pairs.](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | Shankar Iyer et al. | Webpage | [DATA](http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv) |
| [CCQA: A New Web-Scale Question Answering Dataset for Model Pre-Training.](https://arxiv.org/pdf/2110.07731) | Patrick Huber et al. | NAACL 2022 | [DATA](https://github.com/facebookresearch/CCQA) |
| [FEVER: a Large-scale Dataset for Fact Extraction and VERification.](https://aclanthology.org/N18-1074.pdf) | James Thorne et al. | NAACL 2018 | [DATA](https://fever.ai/) |
| [CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims.](https://arxiv.org/pdf/2012.00614v2.pdf) | Thomas Diggelmann et al. | NeurIPS 2020 | [DATA](http://climatefever.ai) |
| [Fact or Fiction: Verifying Scientific Claims.](https://aclanthology.org/2020.emnlp-main.609.pdf) | David Wadden et al. | EMNLP 2020 | [DATA](https://scifact.apps.allenai.org) |
| [SPECTER: Document-level Representation Learning using Citation-informed Transformers.](https://arxiv.org/pdf/2004.07180.pdf) | Arman Cohan et al. | ACL 2020 | [DATA](https://github.com/allenai/specter) |
| [Simple Entity-Centric Questions Challenge Dense Retrievers.](https://arxiv.org/pdf/2109.08535.pdf) | Christopher Sciavolino et al. | EMNLP 2021 | [DATA](https://github.com/princeton-nlp/EntityQuestions) |
| [ArchivalQA: A Large-scale Benchmark Dataset for Open Domain Question Answering over Archival News Collections.](https://arxiv.org/pdf/2109.03438v2.pdf) | Jiexin Wang et al. | Arxiv 2021 | NA |
| [Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval.](https://arxiv.org/pdf/2203.03367) | Dingkun Long et al. | SIGIR 2022 | [DATA](https://github.com/Alibaba-NLP/Multi-CPR) |
| [HOVER: A Dataset for Many-Hop Fact Extraction And Claim Verification.](https://arxiv.org/pdf/2011.03088) | Yichen Jiang et al. | EMNLP 2020 | [DATA](https://hover-nlp.github.io/) |
| [TREC 2021 Deep Learning Track Guidelines.](https://www.microsoft.com/en-us/research/uploads/prod/2022/05/trec2021-deeplearning-overview.pdf) | Nick Craswell et al. | NA | NA |
| [MSMarco Chameleons: Challenging the MSMarco Leaderboard with Extremely Obstinate Queries.](https://ls3.rnet.ryerson.ca/wiki/images/0/09/MSMarco_Chameleons.pdf) | Negar Arabzadeh et al. | CIKM 2021 | [Roff](https://github.com/Narabzad/Chameleons) |


## Libraries
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [RocketQA](https://github.com/PaddlePaddle/RocketQA) | --- | webpage | [Python](https://github.com/PaddlePaddle/RocketQA)|
| [Billion-scale similarity search with GPUs.](https://arxiv.org/pdf/1702.08734) | Jeff Johnson et al. | TBD 2019 | [Python](https://github.com/facebookresearch/faiss) |
| [Pyserini: An Easy-to-Use Python Toolkit to Support Replicable IR Research with Sparse and Dense Representations.](https://arxiv.org/pdf/2102.10073) | Jimmy Lin et al. | Arxiv 2021 | [Python](https://github.com/castorini/pyserini/) |
| [MatchZoo: A Learning, Practicing, and Developing System for Neural Text Matching.](https://arxiv.org/pdf/1905.10289.pdf) | Jiafeng Guo et al. | SIGIR 2019 | [Python](http://www.bigdatalab.ac.cn/matchzoo/#/) |
| [Anserini: Enabling the Use of Lucene for Information Retrieval Research.](https://dl.acm.org/doi/pdf/10.1145/3077136.3080721) | Peilin Yang et al. | SIGIR 2017 | [Java](https://github.com/castorini/anserini) |
| [Tevatron: An Efficient and Flexible Toolkit for Dense Retrieval.](https://arxiv.org/pdf/2203.05765.pdf) | Luyu Gao et al. | Arxiv 2022 | [Python](https://github.com/texttron/tevatron) |
| [Asyncval: A Toolkit for Asynchronously Validating Dense Retriever Checkpoints during Training.](https://arxiv.org/pdf/2202.12510.pdf) | Shengyao Zhuang et al. | SIGIR 2022 | [Python](https://github.com/ielab/asyncval) |
| [Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations.](https://dl.acm.org/doi/pdf/10.1145/3404835.3463238) | Jimmy Lin et al. | SIGIR 2021 | NA
| [OpenMatch: An Open Source Library for Neu-IR Research.](https://arxiv.org/pdf/2102.00166.pdf) | Zhenghao Liu et al. | SIGIR 2021 | [Python](https://github.com/thunlp/OpenMatch)
| [SentEval: An Evaluation Toolkit for Universal Sentence Representations.](https://arxiv.org/pdf/1803.05449.pdf) | Alexis Conneau et al. | Arxiv 2018 | [Python](https://github.com/facebookresearch/SentEval)
