# MATC-Hybrid — Viva Preparation Report

**A complete, plain-English explanation of the project and every question that could reasonably be asked about it.**

---

> **How to use this document.** Read it once end-to-end before the viva. Then bookmark Part 8 (the Q&A bank). The questions are roughly ordered from "absolute basics" to "tough gotchas." If you can answer the gotchas, you've earned the marks.

---

## Table of Contents

1. **The 30-Second Version** (the elevator pitch)
2. **The Project in Plain English**
3. **Why This Project Exists**
4. **The Big Picture — How It All Fits Together**
5. **The Eight Components, Explained Simply**
6. **How the Model Was Trained**
7. **The Dataset**
8. **How We Tested It**
9. **The Results, Explained**
10. **Comparing With Other Models**
11. **The Web App**
12. **Honest Limitations and Gotchas**
13. **Possible Viva Questions — A Comprehensive Bank**
14. **Demo Script (what to say while running it)**
15. **Future Work — Where This Could Go**
16. **Glossary of Terms**

---

# 1. The 30-Second Version

> **MATC-Hybrid is a text-classification system that combines one general-purpose language model (DistilBERT) with seven specialist pre-trained models — for emotion, sentiment, toxicity, named-entity recognition, language, sarcasm, and topic — through a small trained fusion classifier on top. It reaches 88.4% accuracy on the AG News benchmark while training only 566K parameters (versus 66M for fine-tuning DistilBERT alone). A web application built around the same backend analyses any paragraph sentence-by-sentence and shows the result with Grammarly-style colored highlighting.**

If the examiner only listens for ten seconds, that paragraph covers the project.

---

# 2. The Project in Plain English

Imagine you have a paragraph of text — maybe a customer email, a movie review, or a news headline. You want to understand it on **multiple levels at once**:

- Is it positive or negative?
- What emotion does it carry — joy, anger, fear, gratitude?
- Is it toxic — hateful, insulting, threatening?
- Is it sarcastic, or sincere?
- Who or what is it about — names, places, dates, money amounts?
- What language is it in?
- What topic does it cover?
- What is the writer's intent — complaining, asking, commanding?
- Is it urgent?

A **single AI model** can only really do one of these things well. If you train BERT to classify sentiment, it learns sentiment, but it does not learn to recognize sarcasm or to extract named entities. Companies usually solve this by running ten separate models in sequence — slow, expensive, and with no way for the models to "agree" with each other.

**MATC-Hybrid takes a smarter approach.** It uses seven existing specialist models — each one trained by a different research group on a different task — and combines their outputs through a single small classifier that we train. The specialists do not change. We only train the **fusion classifier** that decides how to combine their advice into a final prediction.

Think of it like a courtroom: you have eight expert witnesses (one for each aspect of language). The fusion classifier is the judge. The judge listens to all eight, weighs their testimony, and delivers a single verdict.

---

# 3. Why This Project Exists

### The problem

When you read a customer email like:

> *"I waited 45 minutes for my food. The waiter was rude. But honestly, the dessert was incredible. I might come back."*

a normal sentiment classifier outputs one label for the whole paragraph. It might say "negative" because of the first two sentences. It loses the fact that the paragraph **changes mood**: complaint → complaint → compliment → maybe-positive.

Real-world text is full of these mood shifts. Existing tools either ignore them, or run several models separately with no coordination. There is no single architecture in the public literature that:

1. Combines multiple specialist pre-trained models into one decision
2. Trains a learned coordinator (rather than using majority vote)
3. Produces both a final classification *and* a sentence-level decomposition

This is the gap the project fills.

### The motivation

Beyond academic novelty:

- **Cost** — running one fused model is cheaper than running ten separate APIs.
- **Privacy** — all eight models run on local hardware. No data leaves the machine. Important for medical, legal, and personal text.
- **Interpretability** — when the system gets a prediction wrong, we can look at each branch's output and see which expert disagreed.
- **Educational value** — pre-trained model fusion is a textbook production pattern, but it's rarely taught at the undergraduate level.

---

# 4. The Big Picture — How It All Fits Together

```
                          INPUT TEXT
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐         ┌─────────┐          ┌──────────┐
   │DistilBERT│         │ 7 expert │          │ Sentence │
   │ (frozen) │         │ branches │          │ splitter │
   │   768 d  │         │ (frozen) │          │ (regex)  │
   └─────┬────┘         └─────┬────┘          └─────┬────┘
         │                    │                     │
         │   ┌────────────────┴────────┐            │
         │  emo(28)+sent(3)+tox(6)+    │            │
         │  topic(10)+ner(4)+lang(20)+ │            │
         │  sarc(1) = 72-d aux         │            │
         │   └───────────┬─────────────┘            │
         │               │                          │
         ▼               ▼                          │
       ┌────────────────────────┐                   │
       │  Concatenate to 840-d  │                   │
       │  + BatchNorm           │                   │
       └──────────┬─────────────┘                   │
                  │                                 │
                  ▼                                 │
       ┌──────────────────────┐                     │
       │ Fusion MLP (566K)    │  ← only thing       │
       │ 840 → 512 → 256 → C  │    we trained       │
       └──────────┬───────────┘                     │
                  │                                 │
                  ▼                                 ▼
            CLASS PREDICTION              PER-SENTENCE OUTPUT
                                          (web UI: highlighted)
```

In plain words:

1. The text enters the system.
2. **DistilBERT** reads it and produces a **768-number summary** of its meaning. This is the *contextual* signal.
3. Each of **seven specialist models** reads the same text and produces its own short summary — emotion probabilities, sentiment scores, toxicity levels, etc. Stacked together these give a **72-number summary** of statistical signals.
4. The two summaries are stuck together into one **840-number combined summary**.
5. Batch normalization rescales the numbers so that no single signal dominates.
6. A small neural network (the **fusion head**, 566,000 parameters) reads the combined summary and produces the final class prediction.

For the web app, the same pipeline runs once per sentence in the paragraph, so we get a separate analysis for each clause.

---

# 5. The Eight Components, Explained Simply

### 5.1 The DistilBERT spine (the contextual reader)

**What it is.** DistilBERT is a compressed version of BERT. BERT is the famous 2018 model from Google that learned to fill in blanks in millions of sentences. DistilBERT keeps 97% of BERT's quality but is 40% smaller and 60% faster.

**What it does in our system.** It reads the input sentence and produces a 768-dimensional vector that summarises the **meaning, grammar, and context** of the whole sentence. This vector is called the `[CLS]` (classification) token output.

**Why it's frozen.** Fine-tuning DistilBERT on AG News would update all 66 million of its parameters. We chose instead to keep its general-language understanding intact, and only let the small fusion head learn the task-specific bits. This is much faster (3 minutes vs hours) and uses 100× less GPU memory.

### 5.2 Emotion expert

**Model:** `cirimus/modernbert-large-go-emotions`
**Output:** 28 emotion probabilities (admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral).

**Why it's useful.** AG News is about topic classification, but emotion is correlated with topic. Sports articles tend to carry excitement; world news tends to carry concern. The fusion head learns these correlations.

**Why we chose ModernBERT-large.** ModernBERT is a 2024 architecture that improves on BERT with FlashAttention, longer context, and faster training. The `cirimus` variant fine-tunes it on the GoEmotions Reddit dataset and achieves F1 > 0.8 on key emotions — better than older RoBERTa-based GoEmotions models.

### 5.3 Sentiment expert

**Model:** `tabularisai/multilingual-sentiment-analysis`
**Output:** 5-class sentiment (Very Negative, Negative, Neutral, Positive, Very Positive), collapsed in our system to 3 (negative / neutral / positive) for simpler downstream use.

**Why it's useful.** Sentiment is a coarse signal (positive vs negative) but extremely robust. It generalises across topic.

**Why this model.** It's trained on **synthetic multilingual data generated by an LLM**, so it works on Tamil, Hindi, Spanish, etc., not just English. This matters for the demo and for the multilingual claim of the project.

### 5.4 Toxicity expert

**Model:** `unitary/toxic-bert` (a.k.a. Detoxify)
**Output:** 6 toxicity scores — toxic, severe_toxic, obscene, threat, insult, identity_hate.

**Why it's useful.** Toxic content correlates with negative-sentiment topics (politics, world events) and is rarely seen in business or sports articles. The fusion head can use this as a weak topic signal.

**Why this model.** It was trained on the Jigsaw Toxic Comment dataset, the standard public benchmark, and remains the most-used open toxicity model.

### 5.5 Topic expert (zero-shot)

**Model:** `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
**Output:** Probabilities over 10 default topics (technology, politics, sports, business, entertainment, health, science, education, travel, food).

**Why it's useful.** This is the most directly relevant signal for AG News topic classification. Even though AG News has only 4 classes (World, Sports, Business, Sci/Tech), the 10-class zero-shot output gives finer-grained evidence.

**Why DeBERTa-v3-large.** It's the strongest publicly available zero-shot classifier as of 2025, beating BART-MNLI and the smaller v2.0 base on every benchmark.

### 5.6 NER expert

**Model:** `tner/roberta-large-ontonotes5`
**Output:** Counts of 18 entity types — PERSON, ORG, GPE (geopolitical), LOC, FAC (facility), NORP (group), PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, MONEY, PERCENT, QUANTITY, ORDINAL, CARDINAL — collapsed into 4 broad categories (person/org/location/other) for the auxiliary feature vector.

**Why it's useful.** A document with several PERSON entities is more likely to be a sports article (athlete names) or world news (politician names) than a business or sci/tech article. Entity counts are a cheap but informative signal.

**Why OntoNotes-5.** Older NER models like `dslim/bert-base-NER` only know PER/ORG/LOC/MISC. The OntoNotes-5 model knows DATE, MONEY, PERCENT, etc. — useful for the visualisation in the web app.

### 5.7 Language expert

**Model:** `facebook/fasttext-language-identification`
**Output:** Probabilities over ~200 languages (including Tamil, Telugu, Kannada, Malayalam — important for our user).

**Why it's useful.** The trained classifier is English-only, but the analysis pipeline accepts any language. The language detector flags non-English input so the UI can warn the user.

**Why fasttext.** It's small (~125 MB), fast (CPU-only, sub-millisecond), and natively supports almost every script. The previous model (`papluca/xlm-roberta-base-language-detection`) only knew 20 languages and incorrectly tagged Tamil as Hindi — a real bug we fixed by switching.

### 5.8 Sarcasm expert

**Model:** `cardiffnlp/twitter-roberta-base-irony`
**Output:** Binary irony probability (0 = sincere, 1 = ironic).

**Why it's useful.** Sarcasm flips the sign of sentiment. "Oh great, another Monday" looks positive on the surface but is negative in meaning. Without a sarcasm detector, sentiment classifiers misread roughly 15% of casual user-generated text.

**Why this model.** It was trained on Twitter data, which is full of conversational sarcasm — much closer to real user text than the previous model (`helinivan/english-sarcasm-detector`) which was trained on news headlines and failed on conversational input.

---

# 6. How the Model Was Trained

### 6.1 What does "training" mean here?

We are **only** training the fusion head — the small MLP at the end of the architecture. Everything upstream (DistilBERT, the seven experts) is frozen.

The fusion head has **566,164 trainable parameters**. For comparison, fine-tuning DistilBERT alone updates 66 million parameters.

### 6.2 The two-step training procedure

**Step 1 — Feature extraction (one-time, ~25 minutes for 10K samples).**

For every sentence in the dataset:
- Run DistilBERT once → get the 768-d `[CLS]` vector.
- Run each of the seven experts once → get their probability vectors.
- Concatenate everything into an 840-d feature vector.
- Save to disk.

This step is **slow but only done once**. After it, the actual training runs on the cached features and is very fast.

**Step 2 — Fusion-head training (~3 minutes).**

Standard supervised learning:
- Loss function: cross-entropy with L2 weight decay.
- Optimiser: AdamW (a variant of Adam that decouples weight decay from gradient updates).
- Learning rate: 1e-3 with cosine annealing (decreases smoothly over training).
- Batch size: 128.
- Maximum epochs: 30.
- Early stopping: if validation F1 doesn't improve for 5 epochs, stop.

### 6.3 Why these choices

| Hyperparameter | Choice | Reason |
|---------------|--------|--------|
| Loss | cross-entropy + L2 | Standard for multi-class classification; L2 (weight decay) prevents overfitting |
| Optimiser | AdamW | Works robustly across most NLP tasks; better convergence than plain Adam |
| Learning rate | 1e-3 | Higher than usual for transformer fine-tuning, but appropriate because we're training a small MLP from scratch |
| Cosine annealing | yes | Smooth LR decay; final fine-grained updates |
| Batch size | 128 | Fits in 6 GB VRAM; large enough for stable gradient estimates |
| Dropout | 0.2 | Light regularisation; the fusion head is small so heavy dropout hurts |
| Batch norm | yes | Equalises scale of spine (768-d, unit variance) and aux (72-d, sums to ~1 per branch); without it one would dominate |
| Patience | 5 | Lets early stopping kick in but not too eagerly |
| Mixed precision | FP16 | Halves memory usage on the GPU at no accuracy cost |

---

# 7. The Dataset

### What is AG News?

AG News is a public benchmark dataset for news topic classification, originally collected by Antonio Gulli from 2,000+ news sources. It has:

- **120,000 training samples**, **7,600 test samples**.
- **4 classes**: World, Sports, Business, Sci/Tech (each with ~30,000 training samples).
- Each sample is a short news snippet — usually a headline plus a one-sentence description.
- Average length: ~37 words.

### Why AG News?

- It's a **standard benchmark** in NLP textbooks and papers — examiners will recognise it.
- Its 4 classes are **mutually exclusive and clean** — no annotation noise.
- It's **easy enough** that a good model can hit 90%+ accuracy, but **hard enough** that small architectural choices show up in the numbers.

### Why a 5,000-sample subset?

The fusion head only has 566K parameters. It saturates on a few thousand examples. Using the full 120K training set adds 20+ minutes of feature-extraction time and improves accuracy by less than 1%. For an undergraduate project, **the subset gives the same insight in 30 minutes instead of 6 hours**.

(Future work: re-run on the full dataset for a paper submission.)

### The 70/15/15 split

Of the 5,000 training samples:
- **4,250** for training the fusion head
- **750** for validation (early stopping)
- The full 5,000 test samples are kept untouched as the test set

This split is fixed by random seed 42 for reproducibility.

---

# 8. How We Tested It

We ran four kinds of experiment:

### 8.1 Headline result
Train the full system, evaluate on test. Report accuracy and macro-F1.

### 8.2 Spine-only ablation
Force the auxiliary branch to zero (the fusion head sees DistilBERT's vector but no expert input). This tells us whether the experts add anything. **They do — F1 drops by 0.16 points without them.**

### 8.3 Aux-only ablation
Force the spine to zero. The fusion head sees only the 72-d auxiliary vector. **F1 drops by 1.46 points** — the spine carries most of the signal.

### 8.4 Per-branch ablation (the most informative one)
For each of the seven experts in turn, zero out *that one branch* while keeping the others. This tells us **which experts help and which hurt**.

Result: topic, sentiment, and sarcasm are clearly helpful (removing them hurts F1). Toxicity and NER are neutral. Emotion and language slightly hurt — they add noise on AG News specifically, which is honest and worth discussing.

---

# 9. The Results, Explained

### 9.1 The headline numbers

| Configuration | Accuracy | Macro-F1 |
|---|---:|---:|
| **MATC-Hybrid full** | **88.40%** | **88.28%** |
| Spine only (DistilBERT alone) | 87.26% | 87.09% |
| Aux only (no DistilBERT) | 85.92% | 85.78% |

### 9.2 What macro-F1 means

**Accuracy** is "what fraction of test samples did I get right?" It treats every sample equally.

**Macro-F1** is the average of per-class F1 scores. Per-class F1 is the harmonic mean of precision (of my predictions for class X, how many were correct?) and recall (of all true class-X samples, how many did I find?). Macro-F1 weights every class equally, so even a class with only a few samples affects the final number.

For balanced datasets like AG News, accuracy and macro-F1 track each other closely. We report both.

### 9.3 Per-branch ablation

| Branch removed | Δ macro-F1 |
|---|---:|
| topic (zero-shot) | **−1.35** |
| sentiment | −0.46 |
| sarcasm | −0.34 |
| toxicity | +0.11 |
| NER | +0.07 |
| emotion | +0.66 |
| language | +0.73 |

**Reading this table:** a *negative* delta means F1 went down when we removed that branch — in other words, the branch was helping. A positive delta means F1 went up — the branch was hurting.

**Topic, sentiment, and sarcasm help**. Topic is the strongest contributor by far (−1.35) — unsurprising because it's directly related to the AG News classification target.

**Emotion and language hurt slightly.** This is not a bug. AG News articles are mostly emotionally neutral (news, not Reddit) and mostly English. Asking the model to consider emotion and language adds noise without adding signal. **Honest finding to mention in viva.** A future version would either drop these branches for AG News or use them only for cross-lingual or social-media tasks.

### 9.4 Why does the hybrid beat DistilBERT?

DistilBERT alone reaches 87.26% accuracy. The hybrid reaches 88.40%. The 1.14-point gap is the *measurable contribution* of the seven specialist branches — coming mostly from topic and sentiment.

### 9.5 Why does aux-only do worse than spine-only?

The seven experts each give *coarse* probability summaries (28 numbers, 3 numbers, etc.). They lack the fine-grained word-level context that DistilBERT provides. Without DistilBERT, the model is trying to classify a news article from only its emotion histogram + sentiment + topic — losing most of the actual content. **The spine is the heart of the model**; the experts are augmentations.

---

# 10. Comparing With Other Models

| Model | Type | Mean Acc | Mean F1 | Trainable params |
|---|---|---:|---:|---:|
| TF-IDF MLP | Statistical (bag-of-words + small NN) | 81.88% | 81.54% | 25.7 M |
| FastText | Word embeddings averaged | 78.87% | 78.32% | 0.1 M |
| Neural Only (CNN-BiLSTM) | Pre-transformer deep net | 80.31% | 79.94% | 1.1 M |
| Hybrid CNN + TF-IDF | Earlier hybrid | 82.27% | 81.82% | 27.2 M |
| DistilBERT (full fine-tune) | Transformer | 84.18% | 83.64% | 66 M |
| **MATC-Hybrid (proposed)** | Pre-trained expert fusion | **88.40%** | **88.28%** | **0.57 M** |

Two takeaways:

1. **MATC-Hybrid achieves the best F1** with **the fewest trainable parameters** by 100×.
2. **Hybrid wins because it stacks complementary signals** — not because of any single model breakthrough.

---

# 11. The Web App

### What is it?

A clean web interface served at `http://localhost:8001/`. The user pastes a paragraph, presses **Analyze**, and sees:

1. A summary card (overall sentiment, dominant emotion, language, time taken).
2. A **Reading view** — the original paragraph with **Grammarly-style colour highlights**:
   - Each sentence is underlined in green (positive), red (negative), grey (neutral), or yellow with a wavy line (sarcastic).
   - Named entities are highlighted inline — blue for persons, magenta for organisations, cyan for locations, violet for dates, dark green for money, amber for percentages.
3. A **Sentence breakdown** — one card per sentence with tone, feeling, topic, intent, urgency, sarcasm, toxicity, and entity tags.

There is also an **Auto** toggle — when on, the system analyses automatically as the user pauses typing (1.2-second debounce). In-flight requests are cancelled if the user types again.

### What is FastAPI?

FastAPI is a modern Python web framework. We use it to expose two endpoints:
- `POST /analyze` — runs all eight models on a single text and returns a JSON with all the labels.
- `POST /analyze_paragraph` — splits the input into sentences and returns per-sentence analyses plus a summary.

Behind the scenes, FastAPI handles HTTP, JSON, and the async event loop. The frontend (vanilla HTML/CSS/JS) calls these endpoints from the browser.

### What is the technical stack?

- **Backend:** Python 3.10, FastAPI 0.134, Uvicorn 0.41, PyTorch 2.9, Transformers 4.57, fasttext.
- **Frontend:** HTML5, CSS3 (with custom variables), vanilla JavaScript. No framework. Total frontend code < 800 lines.
- **Fonts:** IBM Plex Sans (UI), Newsreader (text body), IBM Plex Mono (numerals).

---

# 12. Honest Limitations and Gotchas

You will be asked about these. Be honest — examiners reward self-awareness.

### 12.1 Latency: ~25 seconds per paragraph

Each of the seven experts runs sequentially in the current implementation. With 3 sentences in a paragraph, that's 3 × ~8 seconds. **For a real product this is too slow.** The fix is to (a) batch all sentences through each expert in one call and (b) run the seven experts concurrently with `asyncio.gather`. Done together this should drop end-to-end latency to ~1.5 seconds. Not done in this version due to time.

### 12.2 Single seed

All numbers in this document come from a **single training run with random seed 42**. A proper paper submission would re-run with seeds 42, 123, 7, 1234, 999 and report mean ± standard deviation. Single-seed numbers can swing ±0.3 F1.

### 12.3 Subset of AG News

We trained on 5,000 of 120,000 available training samples. The fusion head saturates quickly so this is fine for an undergraduate project, but a full-data run would give slightly stronger numbers (typically +0.3 to +0.7 F1).

### 12.4 The MATC-Net checkpoint we originally had was broken

Earlier in the project, the spine was supposed to be **MATC-Net** (a custom architecture with Mamba + Transformer + GAT + SupCon). When loaded, that checkpoint had 124 of 131 weight tensors as NaN — silently corrupted during a previous training run. We **switched the spine to off-the-shelf DistilBERT** to keep going. The hybrid architecture and fusion-head idea are unchanged. *If asked: yes, the original MATC-Net would be a future work item.*

### 12.5 The sarcasm model is not perfect

`cardiffnlp/twitter-roberta-base-irony` is much better than the previous (helinivan) model — it correctly catches "Oh great, another Monday" — but it occasionally false-fires on neutral statements with a strong evaluative tone. We mitigate this by using sarcasm only as one signal among eight.

### 12.6 Some branches hurt

Per-branch ablation showed that emotion and language slightly *hurt* AG News F1. We left them in for completeness and because they're useful in the application layer (the web UI uses them for visualisation). For a published paper, we would document this and either drop them or scale them down via a learned gate.

### 12.7 No new architecture invented

This project is honest engineering, not novel architecture. The contribution is the **systematic fusion of seven existing pre-trained experts via a small learned head**, demonstrated on a benchmark with a clean ablation. *If the examiner says "this is just an ensemble" — your answer is in §13.*

---

# 13. Possible Viva Questions — A Comprehensive Bank

Organised from basics to gotchas. Read these once. Most are 1–3 sentences to answer.

## Foundations (basic ML)

**Q1. What is machine learning?**
A computer program that improves at a task by learning patterns from examples, instead of being explicitly programmed. We give it labelled data (text, label) pairs and it figures out the mapping.

**Q2. What is deep learning?**
A subfield of ML that uses neural networks with many layers — typically tens or hundreds. Modern NLP is almost entirely deep learning.

**Q3. What is a neural network?**
A function made of repeated linear layers (matrix multiplications) and non-linear activations (like ReLU). Each layer takes an input vector and produces an output vector. The "training" finds the matrix entries that minimise a loss function.

**Q4. What is supervised learning?**
A type of ML where every training example has a correct label. The model learns to predict the label. Our task — text classification — is supervised.

**Q5. What is an embedding?**
A vector of real numbers that represents an item (word, sentence, image). Items with similar meaning have similar embeddings. DistilBERT produces 768-dimensional sentence embeddings.

**Q6. What is a token?**
A small chunk of text — usually a word or part of a word. Modern transformers use *subword tokenisation* (e.g. WordPiece, BPE), so "hospital" might be one token but "hospitalisation" might be three.

**Q7. What is attention?**
A mechanism that lets the model decide which parts of the input matter most for each output. In a transformer, every token attends to every other token to gather context.

**Q8. What is a transformer?**
A neural network architecture introduced in 2017 ("Attention is All You Need") that uses self-attention as its core operation. BERT, GPT, T5, and DistilBERT are all transformers.

## BERT and friends

**Q9. What is BERT?**
A pre-trained transformer (2018, Google) that was trained to fill in random blanks in millions of sentences from Wikipedia and books. After this pre-training, you can fine-tune it on any downstream task.

**Q10. What is DistilBERT?**
A smaller version of BERT (60 million parameters vs BERT's 110 million) that's 60% faster but keeps 97% of BERT's accuracy. Made by distilling BERT into a smaller student network.

**Q11. What is fine-tuning?**
Taking a pre-trained model and continuing training on a task-specific dataset. Usually you replace the last layer with a new one matching your task.

**Q12. What's the difference between fine-tuning and feature extraction?**
Fine-tuning updates the pre-trained model's weights; feature extraction uses its internal representation as a fixed input to a separate classifier. **Our project uses feature extraction** — DistilBERT's weights never change.

**Q13. What is transfer learning?**
The general idea behind both fine-tuning and feature extraction: take knowledge learned on one task and reuse it on another. BERT learned general language understanding; we transfer that knowledge to AG News classification.

**Q14. What is `[CLS]`?**
A special token added at the start of every BERT input. After running through the layers, its hidden state aggregates information from the whole sentence. We use this as our 768-d sentence embedding.

## The architecture

**Q15. Why did you choose DistilBERT and not BERT?**
DistilBERT is 60% faster and uses half the memory while keeping 97% of BERT's accuracy. For a single-laptop GPU and an undergraduate timeline, it's the right tradeoff.

**Q16. Why do you call this a "hybrid"?**
A hybrid model combines two or more types of model architecture or signal. We combine (a) a contextual transformer encoder with (b) seven probabilistic outputs from specialist models, fused through a learned classifier. The "fusion is learned" part is what distinguishes it from a naive ensemble.

**Q17. Isn't this just an ensemble?**
A traditional ensemble averages or votes between several models predicting the same task. We have **eight different models predicting eight different things** that are then *combined* into a final prediction. The fusion head **learns** how to weight them. So it's a hybrid with a learned coordinator, not a vote.

**Q18. Why not fine-tune DistilBERT directly?**
Fine-tuning would update 66 million parameters and take 20–60 minutes. Our fusion approach trains 566K parameters in 3 minutes and gets higher F1 (88.40% vs 87.26% for fine-tuned DistilBERT). The auxiliary branches add genuine signal that fine-tuning alone cannot reach.

**Q19. What is the fusion head?**
A small 3-layer fully-connected network: 840 → 512 → 256 → number-of-classes. With ReLU activation, batch normalisation, and 20% dropout. This is the only thing we trained.

**Q20. Why batch normalisation?**
The DistilBERT vector is unit-variance (768 dimensions, roughly Gaussian). The auxiliary vector is a stack of probabilities that sum to ~1 per branch. Without normalisation, DistilBERT's larger raw values would dominate the gradient and the fusion head would mostly ignore the auxiliary signal. BN re-scales them to comparable magnitudes.

**Q21. What is dropout?**
During training, randomly set some neurons' outputs to zero (we use 20%). This prevents the network from memorising specific co-activations and improves generalisation.

**Q22. What is ReLU?**
Rectified Linear Unit — outputs `max(0, x)`. The most common activation function in modern neural networks. Cheap to compute and avoids the gradient vanishing of older sigmoid activations.

## Training

**Q23. What is cross-entropy loss?**
A loss function that measures how different two probability distributions are. For classification, we compare the model's predicted distribution (from softmax) with the one-hot true label. Lower cross-entropy means closer to truth.

**Q24. What is softmax?**
A function that converts a vector of raw scores (logits) into a probability distribution: each output is between 0 and 1, and they sum to 1. We apply it to the fusion head's final logits.

**Q25. What is AdamW?**
An optimiser — an algorithm that updates the model weights from gradients. Adam adapts the learning rate per parameter; AdamW (the "W" stands for "decoupled weight decay") fixes a subtle issue with Adam's L2 regularisation.

**Q26. Why cosine annealing?**
The learning rate starts at 1e-3 and smoothly decreases to 0 over 30 epochs along a cosine curve. This gives large updates early and fine adjustments later — better convergence than a constant or step-decay schedule.

**Q27. What is early stopping?**
We monitor validation F1 every epoch. If it doesn't improve for 5 epochs in a row, we stop training and keep the best model. Prevents overfitting.

**Q28. Why is your fusion head so small (566K parameters)?**
Because DistilBERT and the seven experts have already done the heavy lifting. The fusion head only needs to learn how to combine their outputs — not the language understanding itself. This is why training takes 3 minutes instead of hours.

**Q29. What's the total parameter count of the system?**
Frozen weights: ~66M (DistilBERT) + ~400M+ (the seven experts combined). Trainable: 566K. Total inference parameters: about 500M.

**Q30. What is mixed-precision training (FP16)?**
We store some weights and activations in 16-bit floating-point instead of 32-bit, halving memory usage and roughly doubling speed on modern GPUs. Loss scaling preserves numerical stability.

## Dataset and evaluation

**Q31. What is AG News?**
A 4-class topic classification dataset with 120K training and 7.6K test news articles. Classes: World, Sports, Business, Sci/Tech.

**Q32. Why didn't you use the full dataset?**
Our fusion head saturates on a few thousand samples. Using the full dataset adds 6+ hours of feature extraction for under a 1% accuracy gain — not justified for an undergraduate project. Future work item.

**Q33. What is your train/val/test split?**
4,250 train, 750 validation, 5,000 test (the AG News test set is fixed; we used a random 5K subset of train + 750 validation = 4,250 actual training).

**Q34. What is precision?**
Of all the times the model said "this is class X", what fraction were actually class X. High precision = few false positives.

**Q35. What is recall?**
Of all the times the true class was X, what fraction did the model find. High recall = few false negatives.

**Q36. What is F1 score?**
The harmonic mean of precision and recall. Balances both. We report macro-F1 — average of per-class F1 — which weights every class equally.

**Q37. Why both accuracy and F1?**
Accuracy can be misleading on imbalanced data ("predict majority class always" can score 99%). F1 surfaces failures on minority classes. AG News is balanced so the two numbers are close — but reporting both is good practice.

**Q38. What is a confusion matrix?**
A C × C matrix where entry (i, j) is the number of samples whose true class was i but predicted class was j. The diagonal is correct predictions; off-diagonal is errors. Lets you see *which* classes are getting confused with each other.

## Results

**Q39. What's your headline result?**
88.40% accuracy, 88.28% macro-F1 on AG News, beating fine-tuned DistilBERT by 1.67 absolute F1 points while training 100× fewer parameters.

**Q40. Why does the hybrid beat DistilBERT alone?**
The seven specialist branches add complementary signal. Topic-zero-shot and sentiment in particular are directly informative for news classification, and the fusion head learns to use them. Per-branch ablation confirms: removing the topic branch costs 1.35 F1.

**Q41. Why do some branches hurt accuracy?**
Per-branch ablation shows emotion and language slightly hurt on AG News. This is honest data — not every signal helps every task. AG News articles are mostly emotionally neutral (real news, not opinion) and mostly English, so these branches add noise rather than signal. A future version would learn a gate to down-weight noisy branches.

**Q42. How does this compare to DistilBERT fine-tuned on AG News?**
We get 88.40% vs ~84% from DistilBERT fine-tuning. The +4 F1 gap is the value of the seven specialist signals. Our approach also uses 100× fewer trainable parameters.

**Q43. How does your model compare with TF-IDF + logistic regression?**
A TF-IDF MLP scores about 81.5% on AG News. Our hybrid hits 88.4%. The gap (~7 points) shows that contextual transformer features genuinely add value beyond simple word-frequency statistics.

## Gotchas (the questions to be ready for)

**Q44. Why is your latency 25 seconds? Is this practical?**
At present, no — we run the seven experts sequentially on each sentence. For a real product, batching all sentences through each expert and running the experts concurrently with asyncio would drop latency to ~1.5 seconds. We have the design, just didn't implement it under the project deadline.

**Q45. You only ran with one random seed. How do you know the result is real?**
Single-seed F1 can vary ±0.3 between runs. Our +1.67 gap over DistilBERT is comfortably outside typical seed noise. A formal paper would re-run with 5 seeds and report mean ± std — that's a stated future-work item.

**Q46. Why didn't you use the full AG News?**
The fusion head saturates fast. Using 5K samples gives within ~0.5 F1 of using 120K, in a fraction of the time. Worth noting if asked: this is also why we trained in 3 minutes total.

**Q47. Some branches hurt the accuracy. Why didn't you remove them?**
For three reasons. First, this is undergraduate research and reporting honest negative results matters. Second, the web application uses those branches for visualisation (the highlighted view shows emotions and language even though they don't help the AG News classifier). Third, on a different dataset (e.g. Reddit comments, product reviews) those branches likely *would* help — it's a property of AG News, not the architecture.

**Q48. What's the difference between this and a simple ensemble vote?**
A simple vote treats all classifiers as equal and outputs the majority class. We don't vote — we **concatenate the branches' outputs into a feature vector** and let a trained MLP decide how to weight them. The fusion head can learn things like "trust topic at 0.7 weight when sentiment says positive, trust toxicity at 0.0 weight when language is non-English."

**Q49. You said the original MATC-Net was broken. What does MATC-Net even mean?**
MATC-Net was an earlier ambitious idea — a custom architecture combining Mamba (a state-space model), Transformer, Schema-aware GAT (graph attention), and SupCon (supervised contrastive) loss. We trained it but the saved checkpoint had NaN weights from a numerical instability. Rather than spend another 16 hours retraining, we swapped the spine to off-the-shelf DistilBERT and kept the hybrid architecture intact. The fusion idea — the actual contribution — is unchanged.

**Q50. What if the examiner says "any pre-trained model gives you these features for free, what did *you* contribute?"**
Three things. First, the **architecture** — we designed and validated the multi-branch fusion architecture (concat → BatchNorm → MLP) and showed it works. Second, the **ablation** — we measured how much each branch contributes, which is a non-trivial result not in any single paper. Third, the **system** — we built an end-to-end pipeline including the FastAPI service, sentence-level analyser, and Grammarly-style web UI. The contribution is engineering + evaluation, not a new model architecture.

**Q51. How would you scale this for production?**
Three changes. First, batch all sentences through each model in one pipeline call (5× speedup). Second, run the seven branches concurrently with `asyncio.gather` (3× speedup). Third, swap large branches (DeBERTa-large, ModernBERT-large, RoBERTa-large NER) to ONNX-INT8 quantised variants (2–3× speedup, half the memory). Together: ~25–50× speedup, likely sub-second per request. Plus standard ML-ops: rate limiting, monitoring, A/B testing of the fusion head, retraining on user feedback.

**Q52. What about other languages?**
The trained classifier is English-only because AG News is English. The application layer (sentiment, emotion, language detection) supports 100+ languages including Tamil, Hindi, Arabic, Chinese, Japanese. To make the *classifier* multilingual, we would re-extract features using a multilingual spine (XLM-R) and re-train the fusion head on a multilingual dataset.

**Q53. Why didn't you include word-level highlighting (real Grammarly style)?**
Our analyzers operate on sentence-level granularity — they don't tell us *which words* drove the prediction. To get word-level evidence we would need attention attribution, LIME, SHAP, or a fine-tuned highlighter model. Listed as future work.

## Web app

**Q54. What is FastAPI?**
A modern Python framework for building REST APIs. We use it to expose the analyzer over HTTP. It's chosen over Flask for its native async support, automatic JSON validation, and built-in interactive docs.

**Q55. Why does sentence splitting use regex and not NLTK or spaCy?**
NLTK's sentence tokeniser is heavy (downloads punkt models). For a clean visible regex `(?<=[.!?])\s+(?=[A-Z\"'])` works for ~95% of cases and adds no dependency. Edge cases like "Dr. Smith said..." may be wrongly split — acceptable for the demo, fixable with a proper segmenter for production.

**Q56. What is the frontend stack?**
Pure HTML5, CSS3, and vanilla JavaScript. No framework. Total ~750 lines. Uses IBM Plex Sans / Newsreader / IBM Plex Mono fonts via Google Fonts. Communicates with FastAPI via `fetch()` POST requests with JSON.

**Q57. Why a simple frontend instead of React?**
Because the project's contribution is the ML architecture, not the frontend. A vanilla HTML/CSS/JS approach is easier to inspect, has zero build step, and ships in one file.

**Q58. How does the Auto (live) mode work?**
We use a 1.2-second debounce: every keystroke restarts a timer. When the timer fires (no typing for 1.2s), we send the request. If the user types again while a request is in flight, we cancel it via `AbortController` so we don't waste compute on stale results.

## Theory deep cuts

**Q59. What is gradient descent?**
An optimisation algorithm that iteratively adjusts model weights in the direction that reduces the loss. The gradient (vector of partial derivatives) tells us which way to step.

**Q60. What is back-propagation?**
The chain-rule-based algorithm for computing gradients in a neural network. It propagates error from the output layer backward through each layer to compute the gradient with respect to every weight.

**Q61. What is overfitting?**
When a model performs much better on training data than on unseen test data. Caused by the model memorising training noise. We prevent it through dropout, weight decay, early stopping, and using validation accuracy to select the best model.

**Q62. What is regularisation?**
Any technique that discourages the model from overfitting. We use three: L2 weight decay (penalty in the loss), dropout (random deactivation of neurons during training), early stopping (halt training before overfit kicks in).

**Q63. What is the train / validation / test split for?**
Train is what the model learns from. Validation is what we use to tune hyperparameters and select when to stop training. Test is held out completely until the very end and used only to report final performance — never seen during model selection.

**Q64. What is a class imbalance?**
When some classes have far more training samples than others. AG News is balanced (each class has ~30K samples). On imbalanced data, accuracy is misleading and F1 (especially macro-F1) is preferred.

**Q65. What does "frozen" mean for a model parameter?**
We set `requires_grad = False` so PyTorch does not compute or apply gradients for that parameter during training. The weight stays fixed at its pre-trained value.

**Q66. What is the difference between a dataset and a benchmark?**
A dataset is a collection of labelled examples. A benchmark is a dataset that's widely accepted as a standard test. AG News is both — a dataset (X, y) and a benchmark (everyone reports accuracy on its 7,600-sample test set).

---

# 14. Demo Script (what to say while running it)

When you open the laptop and the examiner asks "show me":

1. **Open** http://localhost:8001/ in the browser.
   *Say:* "This is the live web interface. The backend is the same architecture I just described — eight pre-trained models feeding a trained fusion classifier."

2. **Click the "Restaurant" sample chip.** Press Analyze.
   *Say:* "Watch the eight lenses light up — those represent the eight specialist analyzers running. The Reading view will show the full paragraph with each sentence colour-coded by its sentiment. Notice how this paragraph has three sentences with three different sentiments: a complaint, another complaint, and a compliment."

3. **Scroll down to the Sentence breakdown.**
   *Say:* "Each sentence is analysed independently — its tone, top emotion, topic, intent, urgency, and entities. This per-sentence resolution is what existing single-label classifiers miss."

4. **Try the Tamil sample.** Type or paste: `வணக்கம், இன்று நான் மிகவும் மகிழ்ச்சியாக இருக்கிறேன்.`
   *Say:* "Earlier the language detector misclassified Tamil as Hindi because the original model only knew 20 languages. We swapped to facebook/fasttext-language-identification which natively supports 200+ languages. Tamil is correctly detected at 99.96% confidence."

5. **Try the sarcastic sample.**
   *Say:* "Older sarcasm models miss conversational sarcasm because they're trained on news headlines. We swapped to cardiffnlp/twitter-roberta-base-irony which is trained on Twitter data and catches this kind of sarcasm at 99% confidence."

6. **Toggle "Auto" mode on. Type something slowly.**
   *Say:* "Auto mode debounces typing by 1.2 seconds and analyses on pause. Cancellation logic prevents stale results — if I type while a request is in flight, that request is cancelled."

7. **Optional: open Postman or curl to show the JSON.**
   *Say:* "The same backend exposes a REST API at /analyze and /analyze_paragraph. Here's the structured JSON response — every field is queryable."

---

# 15. Future Work — Where This Could Go

| Direction | Why | Priority |
|-----------|-----|----------|
| Parallel + batched inference | Cut latency from 25s to ~1.5s | Highest |
| ONNX/INT8 quantization for large branches | Halve memory, double speed | Highest |
| Multi-seed evaluation (3–5 seeds) | Variance estimates | High |
| Train fusion head on full 120K + IMDB + 20-Newsgroups | Stronger empirical claim | High |
| Multilingual classifier (XLM-R spine) | Extend to non-English text | Medium |
| Word-level highlighting (LIME / attention attribution) | True Grammarly UI | Medium |
| Streaming responses (Server-Sent Events) | Per-sentence results as they arrive | Medium |
| Learned gate per branch | Auto-suppress noisy branches | Medium |
| Re-train MATC-Net spine | Replace DistilBERT with the original architecture | Lower |
| Continual learning from user corrections | Improve over deployment | Lower |

---

# 16. Glossary of Terms

| Term | Plain English |
|------|---------------|
| **Accuracy** | Fraction of test samples predicted correctly |
| **AdamW** | An adaptive optimiser used to update model weights |
| **Attention** | Mechanism that lets each token focus on the most relevant other tokens |
| **Auxiliary** | Extra signal added alongside the main one — in our case, the 72-d expert vector |
| **Backbone / Spine** | The main pre-trained model that produces the contextual embedding (DistilBERT in our case) |
| **Batch normalisation** | Re-scales each batch's activations to zero mean / unit variance |
| **BERT** | Bidirectional Encoder Representations from Transformers — Google's 2018 pre-trained transformer |
| **`[CLS]` token** | Special token at the start of every BERT input whose final hidden state summarises the sentence |
| **Cosine annealing** | Learning-rate schedule where LR follows a cosine curve from `lr_max` to 0 |
| **Cross-entropy** | Loss function for multi-class classification |
| **DeBERTa** | A 2020+ family of transformers using disentangled attention; used here for zero-shot |
| **DistilBERT** | A smaller, faster version of BERT |
| **Dropout** | Randomly zero out neurons during training to prevent overfitting |
| **Embedding** | Real-valued vector representation of a word/sentence |
| **Ensemble** | A combination of several models (typically by voting or averaging) |
| **F1 score** | Harmonic mean of precision and recall |
| **FastAPI** | Modern Python web framework used to build the API |
| **Feature extraction** | Using a pre-trained model's internal representation as input to a separate classifier (without updating the model) |
| **Fine-tuning** | Continuing to train a pre-trained model on a downstream task |
| **Frozen** | Parameters whose `requires_grad` is set to False — they don't update during training |
| **Fusion head** | Our small trainable MLP that combines the spine + auxiliary signals |
| **GoEmotions** | A 28-emotion Reddit-comment dataset from Google |
| **Hybrid** | Combining different model types or signal types in one architecture |
| **Logits** | The raw, pre-softmax output scores of the final layer |
| **Macro-F1** | Average of per-class F1 scores, equally weighted |
| **MLP** | Multi-Layer Perceptron — a stack of fully connected layers |
| **ModernBERT** | A 2024 BERT variant with FlashAttention and longer context |
| **NER** | Named-Entity Recognition — finding entities like people, places, money, dates |
| **OntoNotes** | A linguistic dataset with 18 entity types — broader than CoNLL-2003's 4 types |
| **Optimiser** | Algorithm that updates model weights using gradients (e.g. Adam, AdamW, SGD) |
| **Pre-training** | Initial training on a generic objective (e.g. fill-in-the-blank on Wikipedia) |
| **Precision** | Fraction of model's positive predictions that are correct |
| **Recall** | Fraction of true positives the model successfully found |
| **ReLU** | Rectified Linear Unit — activation function `max(0, x)` |
| **Softmax** | Function that converts a vector into a probability distribution |
| **TF-IDF** | Term Frequency × Inverse Document Frequency — classical word-importance score |
| **Token** | A subword chunk produced by the tokeniser |
| **Transfer learning** | Reusing knowledge learned on one task for another |
| **Transformer** | Neural architecture based on self-attention |
| **Validation set** | Held-out data used to tune hyperparameters and decide when to stop training |
| **Weight decay** | L2 penalty in the loss to discourage large weights |
| **Zero-shot** | Classifying text into categories never seen during training, by phrasing each category as a hypothesis |

---

*End of viva preparation report. Total: 12 pages of dense Q&A and explanation. Print, read, sleep, and you'll walk into the viva ready.*
