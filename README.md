
## Basic concepts of LLMs

### Architecture

#### Useful links

* [Blog: Evolution of LLMs](https://www.appypie.com/blog/evolution-of-language-models)
* [Blog: Architecture and components](https://www.appypie.com/blog/architecture-and-components-of-llms)
* [Video: Bert](https://www.bilibili.com/video/BV1PL411M7eQ/?spm_id_from=333.999.0.0)
* [Video: GPT-1, 2, 3](https://www.bilibili.com/video/BV1AF411b7xQ/?spm_id_from=333.999.0.0&vd_source=2b905b37d387b9f810c2a9e64d914140)
* [Zhihu: Why has the Decoder-only arch become the mainstream?](https://www.zhihu.com/question/588325646/answer/3357252612)
#### Papers
* [Casual Decoder only: GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Non-casual Decoder only: Yuan 1.0](https://arxiv.org/pdf/2110.04725.pdf)
* [Encoder-Decoder: Transformer](https://arxiv.org/pdf/1706.03762.pdf)
* [Encoder only: Bert](https://arxiv.org/pdf/1810.04805.pdf)
* [What LM Architecture and Pretraining objective Work Best for Zero-Shot Generalization](https://arxiv.org/abs/2204.05832)

### Tokenizer

#### Useful links
* [Zhihu: Tokenization](https://zhuanlan.zhihu.com/p/626080766)
* [Zhihu: BPE](https://zhuanlan.zhihu.com/p/424631681)
* [Blog: understanding llm tokenization](https://christophergs.com/blog/understanding-llm-tokenization)
* [Video: Build GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
* [Hugging face NLP Course Chapter6](https://huggingface.co/learn/nlp-course/chapter6/1?fw=pt)

#### Papers
* [BPE](https://arxiv.org/abs/1508.07909)
* [WordPiece](https://arxiv.org/abs/1810.04805)
* [Unigram](https://arxiv.org/abs/1804.10959)

#### Others
* [1hr Talk: Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g)
* [Build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5435s)


## LLM4EDA

### Generation view

#### Papers
1. **Code generation**

   1. [Codex](https://arxiv.org/abs/2107.03374)
   2. [AlpaCode](https://arxiv.org/abs/2203.07814)
   3. `(MLCAD 2023)` Chateda: A large language model powered autonomous agent for eda. [[Paper](https://arxiv.org/abs/2308.10204)]
   4. `(23 May 2023 arxiv)` Chipgpt: How far are we from natural language hardware design. [[Paper](https://arxiv.org/pdf/2305.14019.pdf)]
   5. `(ICCAD23 invited)` Verilogeval: Evaluating large language models for verilog code generation. [[Paper](https://arxiv.org/pdf/2309.07544.pdf)][[Code](https://github.com/NVlabs/verilog-eval)]
   6. `(ICCAD23)` Gpt4aigchip: Towards next-generation ai accelerator design automation via large language models. [[Paper](https://arxiv.org/pdf/2309.10730.pdf)]
   7. `(MLCAD23)` Chip-chat: Challenges and  opportunities in conversational hardware design. [[Paper](https://arxiv.org/abs/2305.13243)]
   8. `(8 Nov 2023 arxiv)` Autochip: Automating hdl generation using llm feedback. [[Paper](https://arxiv.org/abs/2311.04887)][[Code](https://github.com/shailja-thakur/AutoChip)]
   9. `(ASP-DAC24)` Rtllm: An open-source benchmark for design rtl generation with large language model. [[Paper](https://arxiv.org/pdf/2308.05345.pdf)][[Code](https://github.com/hkust-zhiyao/RTLLM)]
   10. `(DATE23)` Benchmarking large language models for automated verilog rtl code generation.  [[Paper](https://arxiv.org/pdf/2212.11140.pdf)][[Code](https://github.com/shailja-thakur/VGen)]

2. **Code Verification & Analysis**
   1. `(31 Oct 2023 arxiv)` Chipnemo: Domain-adapted llms for chip design. [[Paper](https://arxiv.org/pdf/2311.00176.pdf)]
   2. `(28 Nov 2023 arxiv)` Rtlfixer: Automatically fixing rtl syntax errors with large language models. [[Paper](https://arxiv.org/abs/2311.16543)]
   3. `(DAC21)` Autosva: Democratizing formal verification of rtl module interactions. [[Paper](https://arxiv.org/abs/2104.04003)]
   4. `(24 Jun 2023 arxiv)` Llm-assisted generation of hardware assertions. [[Paper](https://arxiv.org/abs/2306.14027)]
   5. `(21 Aug 2023 arxiv)` Unlocking hardware security assurance: The potential of llms. [[Paper](https://arxiv.org/abs/2308.11042)]
   6. `(14 Aug 2023 arxiv)` Divas: An llm-based end-to-end framework for soc security analysis and policy-based protection. [[Paper](https://arxiv.org/abs/2308.06932)]
   7. `(2 Feb 2023 arxiv)` Fixing hardware security bugs with large language models. [[Paper](https://arxiv.org/abs/2302.01215)]

3. **Specification Generation**
   1. (24 Jan 2024 arxiv) SpecLLM: Exploring Generation and Review of VLSI Design
Specification with Large Language Model. [[Paper](https://arxiv.org/pdf/2401.13266.pdf)][[Code](https://github.com/hkust-zhiyao/SpecLLM/tree/main)]

### Optimization view

#### Useful links
* [an Introduction to Numerical and Combinatorial Optimization](https://people.rennes.inria.fr/Eric.Fabre/Papiers/NumOpt.pdf)

#### Papers

* `(NIPS2011)` Algorithms for Hyper-Parameter Optimization. [[Paper](https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)]
* `(ACL2023)` Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers. [[Paper](https://arxiv.org/abs/2212.10559)][[Code](https://github.com/microsoft/LMOps/tree/main/understand_icl)]
1. **Combinatorial / Discrete Problem**
   1. `(ICLR24)` `Prompt` `Type1` LARGE LANGUAGE MODELS AS OPTIMIZERS. [[Paper](https://arxiv.org/abs/2309.03409)][[Code](https://github.com/google-deepmind/opro)]
   2. `(ICLR24)` `Bayesian Optimization` `Type3` LARGE LANGUAGE MODELS TO ENHANCE BAYESIAN OPTIMIZATION. [[Paper](https://arxiv.org/abs/2402.03921)][Code](https://github.com/tennisonliu/LLAMBO)]
   3. `(19 Jan 2024 arxiv)` `Evolutionary algorithm` `Type3` A match made in consistency heaven: when large language models meet evolutionary algorithms. [[Paper](https://arxiv.org/abs/2401.10510)]
   4. `(29 Oct arxiv)` `Evolutionary algorithm` `Type3` Large Language Models as Evolutionary Optimizers. [[Paper](https://arxiv.org/abs/2310.19046)]
   5. `(8 Oct arxiv)` `Prompt` `Type1` Towards Optimizing with Large Language Model. [[Paper](https://arxiv.org/abs/2310.05204)]
2. **Numerical / Continuous Problem**
   1. `(ICLR24)` `Prompt` `Type1` LARGE LANGUAGE MODELS AS OPTIMIZERS. [[Paper](https://arxiv.org/abs/2309.03409)][[Code](https://github.com/google-deepmind/opro)]
   2. `(Nature)` `Prompt` `Type1` Mathematical discoveries from program search with large language models. [[Paper](https://www.nature.com/articles/s41586-023-06924-6)]
   3. `(8 Jul arxiv)` `Prompt` `Type3` Large Language Models for Supply Chain Optimization. [[Paper](https://arxiv.org/abs/2307.03875)][[Code](https://github.com/microsoft/OptiGuide)]
   4. `(NIPS23)` `Prompt` `Type1` Using Large Language Models for Hyperparameter Optimization. [[Paper](https://arxiv.org/abs/2312.04528)]
   5. `(19 Jan 2024 arxiv)` `Evolutionary algorithm` `Type3` A match made in consistency heaven: when large language models meet evolutionary algorithms. [[Paper](https://arxiv.org/abs/2401.10510)]
   6. `(22 Nov 2023 arxiv)` `Reinforcement learning` `Type3` Large Language Model is a Good Policy Teacher for Training Reinforcement Learning Agents. [[Paper](https://arxiv.org/abs/2311.13373)][[Code](https://github.com/ZJLAB-AMMI/LLM4Teach)]
   7. `(25 May 2023 arxiv)` `Reinforcement learning` `Type1` Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory. [[Paper](https://arxiv.org/abs/2305.17144)][[Code](https://github.com/OpenGVLab/GITM)]
   8. `(29 Oct arxiv)` `Evolutionary algorithm` `Type3` Large Language Models as Evolutionary Optimizers. [[Paper](https://arxiv.org/abs/2310.19046)]
## Acknowledgment
1. Zhao W X, Zhou K, Li J, et al. A survey of large language models[J]. arXiv preprint arXiv:2303.18223, 2023. [[Paper](https://arxiv.org/abs/2303.18223)][[Code](https://github.com/RUCAIBox/LLMSurvey)]
2. Zhong R, Du X, Kai S, et al. LLM4EDA: Emerging Progress in Large Language Models for Electronic Design Automation[J]. arXiv preprint arXiv:2401.12224, 2023. [[Paper](https://arxiv.org/pdf/2401.12224.pdf)][Code](https://github.com/Thinklab-SJTU/Awesome-LLM4EDA)]