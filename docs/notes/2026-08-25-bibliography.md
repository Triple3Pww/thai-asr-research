# Bibliography — the 2026-08-25 literature sweep

Every paper surfaced by the Consensus MCP sweep (9 queries; 3 at 3-results, 6 at 20-results after
the account was connected) plus the vendor sources. **145 unique papers.**

**This is an index, not a reading list.** Almost none of these have been read past the abstract, and
none has been run on this box. Status lives in `LITERATURE.md`; the argument lives in
`E:/Claude/visualllm-business/docs/superpowers/notes/2026-08-25-speech-chatbot-methodology-and-latency.md`. ⭐ marks the ones worth reading first.

---

## 1. Human turn-taking — the empirical basis of the latency target

- ⭐ [Universals and cultural variation in turn-taking in conversation](https://consensus.app/papers/details/64f15418669651d2a5379f1f3ec10086/) — Stivers et al., 2009, PNAS, 1306 cites. *The ~200 ms figure. Verify whether Thai is in the 10 languages.*
- ⭐ [Timing in turn-taking and its implications for processing models of language](https://consensus.app/papers/details/92dd6a07c94d5753899917607f4e2b15/) — Levinson & Torreira, 2015, Front. Psychol., 552 cites. DOI 10.3389/fpsyg.2015.00731. *Gaps 200 ms, production >600 ms → humans predict.*
- [Speech Rate and Turn-Transition Pause Duration in Dutch and English](https://consensus.app/papers/details/4cc57c471197550fa993a3f6b0bca84d/) — Hoogland et al., 2023, Languages
- [Predicting turn-ends in discourse context](https://consensus.app/papers/details/f3cfe16c8ed459bfabd783d30555a437/) — Corps et al., 2018, Lang. Cogn. Neurosci. DOI 10.1080/23273798.2018.1552008

## 2. Endpointing and turn-end prediction

### Predictive / anticipatory (the family we were missing)
- ⭐⭐ [Endpoint Anticipation for Low-Latency Spoken Dialogue](https://consensus.app/papers/details/d4cfdc216c1d5a1c8ad9c728301fe906/) — Udupa et al., 2026. DOI 10.48550/arxiv.2606.13450. *2.56 s ahead, −505 ms, +28.4% compute, no streaming ASR needed.*
- [Predictive ASR and Turn-taking Prediction at Once](https://consensus.app/papers/details/d440f890ef425ac8afdbfd9aacc6aba0/) — Fukuda et al., 2025, ASRU. DOI 10.1109/asru65441.2025.11434798
- [Predictive Speech Recognition and End-of-Utterance Detection](https://consensus.app/papers/details/195181251b185b738c9b904636e3f10f/) — Zink et al., 2024. DOI 10.48550/arxiv.2409.19990. *EOU 300 ms early.*
- [Projection of Turn Completion in Incremental Spoken Dialogue Systems](https://consensus.app/papers/details/3c893499b91656ecb607c6de14b74679/) — Ekstedt & Skantze, 2021, SIGdial. DOI 10.18653/v1/2021.sigdial-1.45
- [Predictive Turn-Taking: Leveraging Language Models to Anticipate Turn Transitions](https://consensus.app/papers/details/15ba71682f42500187feb1ed768c1309/) — Pinto et al., 2024, RO-MAN. DOI 10.1109/ro-man60168.2024.10731379

### Voice Activity Projection
- ⭐ [Multilingual Turn-taking Prediction Using Voice Activity Projection](https://consensus.app/papers/details/e73d7171630e5a3291f0bfd58286be3c/) — Inoue et al., 2024, 33 cites. *Monolingual models don't transfer; multilingual matches. Explains Smart Turn's Thai failure.*
- [Real-time and Continuous Turn-taking Prediction Using VAP](https://consensus.app/papers/details/6201071779315e14b481558c46742054/) — Inoue et al., 2024, 32 cites. *Real-time on CPU.*
- [Prompt-Guided Turn-Taking Prediction](https://consensus.app/papers/details/858e35485dc05a429cc5c64e4882509d/) — Inoue et al., 2025. DOI 10.48550/arxiv.2506.21191
- [Triadic Multi-party Voice Activity Projection](https://consensus.app/papers/details/b97a17f8a9a75a89a46d416bb8c1fd37/) — Elmers et al., 2025
- [Predicting End-of-turn and Backchannel Based on Multimodal VAP](https://consensus.app/papers/details/8e107f6460a75d9cb2b42c2b37140984/) — Ishii et al., 2025, ICMI. DOI 10.1145/3716553.3750781

### Detection, datasets, classical
- ⭐ [FastTurn: Unifying Acoustic and Streaming Semantic Cues](https://consensus.app/papers/details/75d2d41847075d5b8e613a835c4c1654/) — Wang et al., 2026. DOI 10.48550/arxiv.2604.01897. *Releases a real human-dialogue turn test set.*
- [Speculative End-Turn Detector](https://consensus.app/papers/details/89bfb7f9a02d5967abd5cb86a4ab4c10/) — Ok et al., 2025. DOI 10.48550/arxiv.2503.23439. *First public ETD dataset.*
- [SoulX-Duplug](https://consensus.app/papers/details/ae6926b949f35a04bd7c6b447b87866b/) — Yan et al., 2026. DOI 10.48550/arxiv.2603.14877
- [LLM-Enhanced Dialogue Management for Full-Duplex SDS](https://consensus.app/papers/details/e1ffe0f87647570ca23ef5714868020c/) — Zhang et al., 2025. DOI 10.48550/arxiv.2502.14145. *0.5B semantic VAD.*
- [Online Multimodal End-of-Turn Prediction for Three-party Conversations](https://consensus.app/papers/details/421f058566975e65b28088aaee6e6b32/) — Lee et al., 2024, ICMI. DOI 10.1145/3678957.3685742
- [Improving response timing estimation by reducing ASR delay](https://consensus.app/papers/details/d2f9a8f339e65313af5c045c3049aa2e/) — Sakuma et al., 2023, Interspeech. DOI 10.21437/interspeech.2023-1618
- [Response Timing Estimation using Dialog Act Estimation](https://consensus.app/papers/details/ae02750845b35f6fa18c8095f05dcb5d/) — Sakuma et al., 2022, Interspeech. DOI 10.21437/interspeech.2022-746
- [Optimizing the turn-taking behavior of task-oriented spoken dialog systems](https://consensus.app/papers/details/d44ce117a87957ee9c5ef2343b928809/) — Raux & Eskenazi, 2012, 59 cites. DOI 10.1145/2168748.2168749. *FSTTM, deployed on CMU Let's Go.*
- [Towards Deep End-of-Turn Prediction for Situated Spoken Dialogue](https://consensus.app/papers/details/635b787751cc5c08a2cc143cf60d3624/) — Maier et al., 2017, 44 cites

## 3. Architecture — cascaded, end-to-end, hybrid

### The self-hosting question
- ⭐⭐ [Building Enterprise Realtime Voice Agents from Scratch](https://consensus.app/papers/details/e7579079fbe056d5aebd77cad3ecada0/) — Qiu et al., 2026. DOI 10.48550/arxiv.2603.05413. ***"Fully self-hosted end-to-end solutions are not yet available."** Cascade TTFA 755 ms on cloud parts.*
- ⭐ [Discourse-Aware Dual-Track Streaming Response (DDTSR)](https://consensus.app/papers/details/7b7ba093454d5ef1bbff56260b5b7dd7/) — Liu et al., 2026. DOI 10.48550/arxiv.2602.23266. *19–51% latency cut on OUR architecture.*
- ⭐ [A Full-duplex Speech Dialogue Scheme Based On LLMs](https://consensus.app/papers/details/56cc7044befe5e86a4e43c2e6bddc19d/) — Lu et al., 2024, 47 cites. DOI 10.48550/arxiv.2405.19487. ***<500 ms in >50% of interactions at 8B.***
- [RelayS2S: Dual-Path Speculative Generation](https://consensus.app/papers/details/648b0f4ff8ce5e578af42c35b06680a6/) — Mai, 2026. DOI 10.48550/arxiv.2603.23346
- [Style-Talker](https://consensus.app/papers/details/6314605e46a75215b96f102ed934dcbc/) — Li et al., 2024. *Runs ASR after playback starts; >50% faster.*

### Making a cascade full-duplex
- [FireRedChat](https://consensus.app/papers/details/e68f3925abf45c1e8631737ad9b48703/) — Chen et al., 2025, 21 cites. DOI 10.48550/arxiv.2509.06502
- [FlexDuo](https://consensus.app/papers/details/16853c2203535c968d0924cc2da760e6/) — Liao et al., 2025, 17 cites. DOI 10.48550/arxiv.2502.13472. *−24.9% false interruptions.*
- [DuplexCascade: VAD-Free Cascaded ASR-LLM-TTS with Micro-Turns](https://consensus.app/papers/details/24138a2b282f5ad59a86b5a9f10a78b3/) — Yang et al., 2026. DOI 10.48550/arxiv.2603.09180
- [Unit-Based Agent for Semi-Cascaded Full-Duplex Dialogue](https://consensus.app/papers/details/510a651fcc0c517f9e245e3422897f79/) — Yu et al., 2026. DOI 10.48550/arxiv.2601.20230

### End-to-end speech-to-speech
- [Moshi](https://consensus.app/papers/details/2ad46951714e5087a26e987c37de2165/) — Défossez et al., 2024, 658 cites. DOI 10.48550/arxiv.2410.00037. *160 ms theoretical / 200 ms practical.*
- [Voila](https://consensus.app/papers/details/faf20e0b5cf15bf9a3d8483e8c2a6ea0/) — Shi et al., 2025. DOI 10.48550/arxiv.2505.02707. *195 ms.*
- [MinMo](https://consensus.app/papers/details/e4f4a079aaf95462a73b181c192b625a/) — Chen et al., 2025, 100 cites. DOI 10.48550/arxiv.2501.06282. *Full-duplex ~800 ms in practice.*
- [SALMONN-omni](https://consensus.app/papers/details/2a655eb6644d579b9e06fe882ee7c841/) — Yu et al., 2025, 32 cites. DOI 10.48550/arxiv.2505.17060
- [BayLing-Duplex](https://consensus.app/papers/details/19d3a7e0abf75861aa6d432a56131b70/) — Fang et al., 2026. DOI 10.48550/arxiv.2606.14528
- [PersonaPlex](https://consensus.app/papers/details/b5235b923aa458e98f0128f16bd3b7d1/) — Roy et al., 2026, 38 cites. DOI 10.48550/arxiv.2602.06053. *Multi-role customer service.*
- [DuplexSLA](https://consensus.app/papers/details/0e0d0c770e4d585086cfb56f92150cf3/) — Zhang et al., 2026. DOI 10.48550/arxiv.2605.20755. *Tool calling without halting audio.*
- [DuplexMamba](https://consensus.app/papers/details/c486ad42588c5a14b942f6a6f65fc496/) — Lu et al., 2025. DOI 10.48550/arxiv.2502.11123
- [Efficient and Direct Duplex Modeling for S2S](https://consensus.app/papers/details/1f557e2816e858aba75608f8de196351/) — Hu et al., 2025, 34 cites. DOI 10.48550/arxiv.2505.15670
- [Open Full-duplex Voice Agent with S2S LM](https://consensus.app/papers/details/25ed47de89165c3aa56dd38d8b3fe4d4/) — Casanova et al., 2025, ASRU. DOI 10.1109/asru65441.2025.11434669
- [F-Actor: Controllable Conversational Behaviour](https://consensus.app/papers/details/4dfcae19f0d4544b89fe301de0f6c497/) — Zufle et al., 2026. DOI 10.48550/arxiv.2601.11329
- [SLAM-Omni](https://consensus.app/papers/details/e36336a44a4d56c8b0d25f66742435eb/) — Chen et al., 2024, 78 cites. DOI 10.18653/v1/2025.findings-acl.115
- [Baichuan-Audio](https://consensus.app/papers/details/200db9430c43514eacf41a2b6d6b1ece/) — Li et al., 2025, 86 cites. DOI 10.48550/arxiv.2502.17239
- [VITA-Audio](https://consensus.app/papers/details/7db085ef069c5c6e98087bc4d9aa5342/) — Long et al., 2025, 38 cites. DOI 10.48550/arxiv.2505.03739
- [PSLM: Parallel Generation of Text and Speech](https://consensus.app/papers/details/0272fc72fe7c517d9371b60837538b91/) — Mitsui et al., 2024. DOI 10.48550/arxiv.2406.12428
- [Chain-of-Thought Reasoning in Streaming Full-Duplex SDS (SCoT)](https://consensus.app/papers/details/cfacc0a996e1553db00918417303ae72/) — Arora et al., 2025. DOI 10.48550/arxiv.2510.02066
- [Stream RAG](https://consensus.app/papers/details/0518af373aad5e999a1ad58c40bb38bd/) — Arora et al., 2025, 22 cites. DOI 10.48550/arxiv.2510.02044. *Predicts tool queries during user speech.*

### Deployed systems and benchmarks
- [Duplex Conversation](https://consensus.app/papers/details/c9811636cfa857fb894f64adebf0dbb7/) — Lin et al., 2022, KDD, 40 cites. DOI 10.1145/3534678.3539209. *Alibaba production; −50% latency.*
- [Toward Low-Latency E2E Voice Agents for Telecommunications](https://consensus.app/papers/details/34ff94ba53045253b0bce906eec2b4de/) — Ethiraj et al., 2025. DOI 10.48550/arxiv.2508.04721
- ⚠️ [τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains](https://consensus.app/papers/details/5113ffd1552f55af9903be981087bdbd/) — Ray et al., 2026. DOI 10.48550/arxiv.2603.13686. *Voice agents keep only 30–45% of text capability.*
- [VoiceAgentBench](https://consensus.app/papers/details/30e51574d9885cf690097328caa8b94b/) — Jain et al., 2025. DOI 10.48550/arxiv.2510.07978. *ASR→LLM beats end-to-end on agentic tasks.*
- [SELMA](https://consensus.app/papers/details/05328e3e0d795446a3abd34d6014f301/) — Wagner et al., 2025, ICASSP. DOI 10.1109/icassp49660.2025.10890139
- [Open-Source Full-Duplex Conversational Datasets](https://consensus.app/papers/details/14c34e6536ce5943a3ea16d3b3fdf406/) — Zhou et al., 2025. DOI 10.48550/arxiv.2509.04093

### Perceived latency (cheap wins)
- [Improving Impressions of Response Delay in AI-based SDS](https://consensus.app/papers/details/30996829cf245ab09a5e82a782c497f0/) — Asaka et al., 2024, RO-MAN. DOI 10.1109/ro-man60168.2024.10731216. *Filler insertion +1.1 Likert.*
- [Mitigating Response Delays with LLM-powered Intelligent Virtual Agents](https://consensus.app/papers/details/b43d7ecde43d59b79778c4bfb14e834d/) — Maslych et al., 2025, CUI, 29 cites. DOI 10.1145/3719160.3736636
- ⚠️ [The Influence of Response Time and Feedback Type on Voice Interaction UX among Older Adults](https://consensus.app/papers/details/718332c4aea552de8b7dc8a29fb356ce/) — Cao et al., 2024, IJHCI. DOI 10.1080/10447318.2024.2348225. ***Optimal dialogue response 600–1500 ms** — the one result arguing against the 500 ms target.*
- [Human Latency Conversational Turns for Spoken Avatar Systems](https://consensus.app/papers/details/57fb64e20d755774ad7d7938ec642248/) — Jacoby et al., 2024, 14 cites. DOI 10.48550/arxiv.2404.16053. *GPT-4 reconstructs a dropped final word >60%.*

## 4. Streaming ASR — the A2 recipes

- ⭐ [GigaAM: Efficient Self-Supervised Learner for Speech Recognition](https://consensus.app/papers/details/5677ea91b84250e0a5c4db3828da465e/) — Kutsakov et al., 2025. DOI 10.48550/arxiv.2506.01192. *Dynamic chunk-size sampling → both modes; MIT.*
- ⭐ [Context-Aware Dynamic Chunking for Streaming Tibetan Speech Recognition](https://consensus.app/papers/details/e21bc206f99252ffacc3a1a9a9b10852/) — Wang et al., 2025. DOI 10.48550/arxiv.2511.09085. *−48.15% relative vs fixed chunk.*
- ⭐ [Dynamic Chunk Convolution for Unified Streaming and Non-Streaming Conformer ASR](https://consensus.app/papers/details/f41a8dbaaeeb56849a2d9d8c3f19017c/) — Li et al., 2023, ICASSP. DOI 10.1109/icassp49357.2023.10097062. *Prices the streaming penalty: 41.7% → 16.7%.*
- ⭐ [XLSR-Transducer: Streaming ASR for Self-Supervised Pretrained Models](https://consensus.app/papers/details/d55d3e76b1fb59ea980791b0be66b358/) — Kumar et al., 2024, ICASSP. DOI 10.1109/icassp49660.2025.10888110
- [Reducing the Offline-Streaming Gap for Unified ASR Transducer](https://consensus.app/papers/details/69c2e21268665e24b920bdef931cc11e/) — Andrusenko et al., 2026
- [Simul-Whisper: Attention-Guided Streaming Whisper](https://consensus.app/papers/details/5e0a42f0fb4650ada292d1c51c06b954/) — Wang et al., 2024, Interspeech. DOI 10.21437/interspeech.2024-1814. *Streaming Whisper, no fine-tuning.*
- [Attention-Free Dual-Mode ASR with Latency-Controlled Selective State Spaces](https://consensus.app/papers/details/4827fec6b06558c7a0857a6a6e5e34c2/) — Moriya et al., 2025, Interspeech. DOI 10.1109/icassp49660.2025.10889802
- [Advancing Streaming ASR with Chunk-wise Attention and Trans-chunk Selective State Spaces](https://consensus.app/papers/details/089101a759865326901279616acd7fc3/) — Mimura et al., 2025, ICASSP
- [Chunk-wise Attention Transducers (CHAT)](https://consensus.app/papers/details/1c188f2d96025ea881898156b659afb0/) — Xu et al., 2026. DOI 10.48550/arxiv.2602.24245
- [Unified Streaming and Non-streaming Two-pass E2E Model (U2++)](https://consensus.app/papers/details/541fa520a0ad5fada172afefaba3a9a2/) — Zhang et al., 2020, 95 cites. DOI 10.48550/arxiv.2012.05481
- [WeNet](https://consensus.app/papers/details/50acf01d4e2d52d7bf3dd04bb449c173/) — Yao et al., 2021, 328 cites. DOI 10.21437/interspeech.2021-1983
- [Conv-Transformer Transducer](https://consensus.app/papers/details/7d12a1cea1a7589aac9fc08fb4659d7b/) — Huang et al., 2020, 53 cites. *140 ms look-ahead.*
- [Chunked Attention-Based Encoder-Decoder Model](https://consensus.app/papers/details/7a60e1ecdd7c56e591ce57f6f4f55d84/) — Zeineldeen et al., 2023, ICASSP. DOI 10.1109/icassp48485.2024.10446035
- [Streaming Chunk-Aware Multihead Attention (SCAMA)](https://consensus.app/papers/details/bd364696c4ef5382993b7aa0e0ebbbca/) — Zhang et al., 2020. DOI 10.21437/interspeech.2020-1972
- [MFLA: Monotonic Finite Look-ahead Attention](https://consensus.app/papers/details/9a58d69f9c585531a3c17f8ee7474ab2/) — Xia et al., 2025. DOI 10.48550/arxiv.2506.03722
- [Streaming Speech Recognition with Decoder-Only LLMs and Latency Optimization](https://consensus.app/papers/details/ea8566721c865577837d80f41630df87/) — Wan et al., 2026. DOI 10.48550/arxiv.2601.22779
- [Efficient Streaming LLM for Speech Recognition (SpeechLLM-XL)](https://consensus.app/papers/details/b7c9df059a8c5279950c2f9fd16a180c/) — Jia et al., 2024, ICASSP. DOI 10.1109/icassp49660.2025.10890853
- [Time-Shifted Contextual Attention and Dynamic Right Context Masking](https://consensus.app/papers/details/460ea39b8d275fa892c31644216a7ab2/) — Le et al., 2024. DOI 10.21437/interspeech.2024-1971
- [Attention Based On-Device Streaming Speech Recognition (MoChA)](https://consensus.app/papers/details/f059d40c328b538eae154cab8df352e3/) — Kim et al., 2019, ASRU, 61 cites. DOI 10.1109/asru46091.2019.9004027
- ⚠️ [Measuring the Accuracy of Automatic Speech Recognition Solutions](https://consensus.app/papers/details/eff65a8166435129a9cd6a21da527c39/) — Kuhn et al., 2023, ACM TACCESS, 65 cites. *Streaming measures worse than batch across 11 services.*

### Low-resource ASR generally
- [Improving ASR Performance for Low-Resource Languages With Self-Supervised Models](https://consensus.app/papers/details/f0c4b0dbfe125db7a5d8ed21179339c9/) — Zhao et al., 2022, 105 cites. DOI 10.1109/jstsp.2022.3184480
- [Exploration of Whisper fine-tuning strategies for low-resource ASR](https://consensus.app/papers/details/62da781892805ef799b3a2bc8e8ae7a3/) — Liu et al., 2024, 60 cites. DOI 10.1186/s13636-024-00349-3
- [Efficiently Fusing Pretrained Acoustic and Linguistic Encoders](https://consensus.app/papers/details/e29e57b67b2352ad9ab61fd73c17c1d4/) — Yi et al., 2021, 44 cites. DOI 10.1109/lsp.2021.3071668
- [TAML-Adapter](https://consensus.app/papers/details/b8befdcb6455565fa08999217409bfdc/) — Liu et al., 2025, IEEE SPL. DOI 10.1109/lsp.2024.3525399
- [WenetSpeech-Yue: Large-scale Cantonese Corpus](https://consensus.app/papers/details/fee2af60da2757f7829ef9ffbe03bd80/) — Li et al., 2025, 19 cites. DOI 10.48550/arxiv.2509.03959. *Template for what a Thai equivalent would look like.*

---

## 5. 🇹🇭 THAI — the language-specific corpus

### Voice agents / dialogue (the whole set — note how short it is)
- ⭐ [Thai-Duplex-Bench](https://consensus.app/papers/details/7e6b12e0fc435f7f97bd27f66cb915cb/) — Popit et al., 2026, JCSSE. DOI 10.1109/jcsse68839.2026.11596692. *GPT-Realtime-Mini 1.5 s vs Gemini Live 4.2 s; cascaded baseline 97.0% strict pass.*
- ⭐ [MANJU: Multi-Agent Framework for Natural Just-in-Time Understanding in Thai](https://consensus.app/papers/details/25a0cd0497005a569f0d010ae9476767/) — Koseeyaumporn et al., 2026, JCSSE. DOI 10.1109/jcsse68839.2026.11596791. ***The only published Thai voice-to-voice loop: 1.8 s first audio, 6.07 s end-to-end.***
- ⭐ [Thai Semantic End-of-Turn Detection for Real-Time Voice Agents](https://consensus.app/papers/details/5a808720fa0e557cab083e838c7726a3/) — Popit et al., 2025, ICSEC. DOI 10.1109/icsec67360.2025.11298110. *arXiv 2510.04016. F1 0.881 @ 110 ms.*

### Thai ASR
- ⭐ [Typhoon ASR Real-time](https://consensus.app/papers/details/5f6dff9194fd55329698dad4e71f4cd1/) — Sirichotedumrong et al., 2026. DOI 10.48550/arxiv.2601.13044. *Normalisation ≈ scaling; Isan curriculum; releases Typhoon ASR Benchmark.*
- ⭐⭐ [Towards Building Speech LLMs for Multitask Understanding in Low-Resource Languages](https://consensus.app/papers/details/adc343282da2552c9baad38380c177c7/) — Shao et al., 2025. DOI 10.48550/arxiv.2509.14804. ***XLSR-Thai: first Thai SSL encoder, 36,000 h. Plus Thai-SUP (>1,000 h SLU).***
- [Weakly Supervised Data Refinement... Efficient Thai LLM-based ASR (EThai-ASR)](https://consensus.app/papers/details/f1e7a2a1987a530394d1f121eb3dc9d6/) — Shao et al., 2025. DOI 10.48550/arxiv.2505.22063
- [Thonburian Whisper](https://consensus.app/papers/details/1a287b185eea51e98526791a035d034e/) — Aung et al., 2024. *WER 5.49% large on CommonVoice 13.*
- [Thai Wav2Vec2.0 with CommonVoice V8](https://consensus.app/papers/details/39ed2cf248a55c49828e4b4e20575b78/) — Phatthiyaphaibun et al., 2022. DOI 10.48550/arxiv.2208.04799
- ⚠️ [Comprehensive Benchmarking and Analysis of Open Pretrained Thai ASR Models](https://consensus.app/papers/details/130ea3a050aa5c41b8fbe92625580a8d/) — Tipakasorn et al., 2024, O-COCOSDA. DOI 10.1109/o-cocosda64382.2024.10800399. *Big read-vs-spontaneous gap.*
- [Adaptive pruning and SVD-enhanced acoustic modeling for low-resource Thai](https://consensus.app/papers/details/cb7b74c7dd8e5715a9f0e176f591af81/) — Zhu et al., 2025. DOI 10.1117/12.3090502
- [Comparative Evaluation of Thai STT Models for Beverage Ordering](https://consensus.app/papers/details/8f7ac48e94455f93902fb79aaa1eaf00/) — Charoenpool et al., 2026. DOI 10.1109/ectidamtncon67592.2026.11460032
- [Two-Stage Comparative Study of Thai-Suitable Speech Models for Mispronunciation Detection](https://consensus.app/papers/details/123a0c72cd0d50e3a3fdee57daafe185/) — Eiamin et al., 2026, JCSSE. DOI 10.1109/jcsse68839.2026.11597078
- [Using ASR to Assess Thai Speech Language Fluency in MoCA](https://consensus.app/papers/details/0fd5257fbd175b249571be72e5e47599/) — Kantithammakorn et al., 2022, Sensors, 22 cites. DOI 10.3390/s22041583 · [thesis version](https://consensus.app/papers/details/2fd4fa46c71f5df09b26438c13309c02/)

### Thai speech corpora — ⭐⭐ the corpus question
- ⭐⭐ [The Development of a Thai Telephone Conversational Speech Corpus](https://consensus.app/papers/details/f2de4d273a2258c4b525fe367ea3df5e/) — Thatphithakkul et al., 2023, O-COCOSDA. DOI 10.1109/o-cocosda60357.2023.10482925. ***37h28m REAL 8 kHz agent↔customer calls, 4,383 customers. Top-5 words are final particles. CHASE AVAILABILITY.***
- ⭐ [GigaSpeech 2](https://consensus.app/papers/details/e94736dedc655416bb0afce4ec5de2ef/) — Yang et al., 2024, 58 cites. DOI 10.48550/arxiv.2406.11546. *30,000 h incl. Thai; −25–40% WER vs Whisper-large-v3.*
- [Thai Dialect Corpus and Transfer-based Curriculum Learning](https://consensus.app/papers/details/d7d4a79a5bf35a2caf78e81f5240ae41/) — Suwanbandit et al., 2023, Interspeech, 18 cites. DOI 10.21437/interspeech.2023-1828. *700 h Thai-central.*
- [Thai-Dialect: Low Resource Thai Dialectal Speech to Text Corpora](https://consensus.app/papers/details/373167870d335bdbbe6c6bb8d26bdbf1/) — Suwanbandit et al., 2023, ASRU. DOI 10.1109/asru57964.2023.10389792
- [Improving Thai-dialect ASR with curriculum learning](https://consensus.app/papers/details/1ac11016e5ac5b52b112798d97d4cb4f/) — Suwanbandit, 2024. DOI 10.58837/chula.the.2024.291
- [LOTUS-TRD: A Thai Regional Dialect Speech Corpus](https://consensus.app/papers/details/69202d7fdb895eba8731f099f13986b9/) — Thatphithakkul et al., 2024. DOI 10.1109/o-cocosda64382.2024.10800335. *180 h, CC-BY-SA 4.0.*
- [PaSCoNT — Parallel Speech Corpus of Northern-central Thai](https://consensus.app/papers/details/e159860f8cac55119a6d6259072f768d/) — Taerungruang et al., 2024, Comput. Speech Lang. DOI 10.1016/j.csl.2024.101692
- [The Development of Northern Thai Dialect Speech Recognition System](https://consensus.app/papers/details/0c26be5a5dba5aaa874e13288b379c67/) — Bootkrajang et al., 2025, Applied Sciences. DOI 10.3390/app16010160
- [Ubiquitous Computing Framework for Lanna Thai Dialect](https://consensus.app/papers/details/c41839e5c77f5d3b89c60a172da3d941/) — Nuankaew et al., 2025, IJACSA. DOI 10.14569/ijacsa.2025.0161225
- ⭐ [THAI-SER: Thai speech emotion recognition corpus](https://consensus.app/papers/details/177c42c9934e584c8cdddff87d2e3d3c/) — Wongpithayadisai et al., 2025, LREV. DOI 10.1007/s10579-026-09932-w. *41.6 h, 200 professional actors, CC-BY-SA 4.0.* · [thesis version](https://consensus.app/papers/details/770d004cae3c548e82b69a5dfe35da54/)

### Thai TTS
- ⭐ [JaiTTS: A Thai Voice Cloning Model](https://consensus.app/papers/details/94b8d8d77c615a2395581c0cf29db13b/) — Karnjanaekarin et al., 2026. DOI 10.48550/arxiv.2604.27607. ***VoxCPM-based. CER 1.94% vs human 1.98%.***
- [ThonburianTTS](https://consensus.app/papers/details/0360ecde03b45612a3d9bce2903d4bcd/) — Aung et al., 2025, iSAI-NLP. DOI 10.1109/isai-nlp66160.2025.11320472
- [Multi-Speaker Thai Speech Synthesis Using Transfer Learning](https://consensus.app/papers/details/bed850c423395a0a8667d974b2168430/) — Charoenrattana et al., 2026, ECTI-CIT. DOI 10.37936/ecti-cit.2026202.265851. *Tonal source languages transfer better.*
- [Empowering Global Voices: Phoneme-Tone Adaptive Speech Synthesis](https://consensus.app/papers/details/12c481bd8ff8576f8554e37b1249a640/) — Geng et al., 2025. *Thai as the case study.*
- [Low-Resource Multilingual and Zero-Shot Multispeaker TTS](https://consensus.app/papers/details/6ef483f6678e52598ee48ea09b612556/) — Lux et al., 2022, 31 cites
- ⭐ [An Ultra-Low Latency, End-to-End Streaming Speech Synthesis Architecture](https://consensus.app/papers/details/8f56e27e4c1f59e58b25cbea6b0d2b61/) — Su et al., 2026. DOI 10.48550/arxiv.2604.12438. ***TTFB 48.99 ms**, validated on Malay.*

### Thai LLM
- [Typhoon: Thai Large Language Models](https://consensus.app/papers/details/93e84386406b506a8a82a0e2c7c2b460/) — Pipatanakul et al., 2023, 35 cites. DOI 10.48550/arxiv.2312.13951. *2.62× more efficient Thai tokenisation.*
- [OpenJAI-v1.0](https://consensus.app/papers/details/4cfd394c97d757a7990036fd6cc4f69a/) — Trakuekul et al., 2025. DOI 10.48550/arxiv.2510.06847. *Qwen3-14B + tool use.*
- [SiamGPT: Quality-First Fine-Tuning for Stable Thai Text Generation](https://consensus.app/papers/details/9fb59b63bf9455c29467e1d8076a276c/) — Pairatsuppawat et al., 2025. DOI 10.48550/arxiv.2512.19455
- [OpenThaiGPT 1.5](https://consensus.app/papers/details/ca497a028ad5550aa73211647abba413/) — Yuenyong et al., 2024. DOI 10.48550/arxiv.2411.07238 · [1.6 and R1](https://consensus.app/papers/details/f2e7749078e5553ebeeed0144be5e437/) — 2025. DOI 10.48550/arxiv.2504.01789
- [WangchanThaiInstruct](https://consensus.app/papers/details/e11b62433e425f9b8dfeda99ef0a5e3c/) — Limkonchotiwat et al., 2025. DOI 10.48550/arxiv.2508.15239. *Native supervision beats translated.*
- [Seed-Free Synthetic Data Generation... A Case Study in Thai](https://consensus.app/papers/details/d908bf2fb22e51cebaa92e97a5160771/) — Pengpun et al., 2024, ACL SRW. DOI 10.18653/v1/2024.acl-srw.38
- [Thai-H6 and ThaiCLI benchmarks](https://consensus.app/papers/details/11202f563a495f9e852333d02c46fea2/) — Kim et al., 2024. DOI 10.48550/arxiv.2410.04795
- ⚠️ [Assessing Thai Dialect Performance in LLMs](https://consensus.app/papers/details/89697de096ae5429b630edc9c8590fbf/) — Limkonchotiwat et al., 2025. DOI 10.48550/arxiv.2504.05898. *Big drop on Isan/Southern.*
- [SeaLLMs](https://consensus.app/papers/details/ecadb3e18b165af8b5cc9069cef99056/) — Nguyen et al., 2023, 149 cites. DOI 10.48550/arxiv.2312.00738
- [SeaLLMs-Audio](https://consensus.app/papers/details/7ebe3a7612065a41a9e0ea3fa7910a1c/) — Liu et al., 2025. DOI 10.48550/arxiv.2511.01670. *First SEA audio-LLM incl. Thai.*
- [MERaLiON-AudioLLM](https://consensus.app/papers/details/803103bb34115007a9c6f362f5414614/) — He et al., 2025, ACL demo. DOI 10.18653/v1/2025.acl-demo.3
- [BURMESE-SAN](https://consensus.app/papers/details/94b147279e0c5edbbd0a9b5ac5737016/) — Aung et al., 2026. DOI 10.48550/arxiv.2602.18788. *Regional comparison point.*

### Thai text processing — word/sentence segmentation (feeds `loop/clause.py`)
- ⭐ [Thoughts on word and sentence segmentation in Thai](https://consensus.app/papers/details/a942a0209c705f41bbbdacdf69f080d7/) — Aroonmanakun, 2007, 50 cites. DOI 10.58837/chula.res.2007.92. ***"Discourse should be seen as a combination of clauses rather than sentences."***
- ⭐ [A Word Labeling Approach to Thai Sentence Boundary Detection and POS Tagging](https://consensus.app/papers/details/86eab23472265fa89ce290ac8d0ca2b7/) — Zhou et al., 2016. *Space is not a reliable boundary marker.*
- [Semi-supervised Thai Sentence Segmentation Using Local and Distant Word Representations](https://consensus.app/papers/details/3a8d792d263853cf98a76ff5471f1ead/) — Saetia et al., 2019, TALLIP. DOI 10.1145/3389037
- [BoydCut: Bidirectional LSTM-CNN Thai Sentence Segmenter](https://consensus.app/papers/details/7f317b547eba5a7fae42800404771454/) — Sirirattanajakarin et al., 2020. DOI 10.1109/ibdap50342.2020.9245454
- [Maximum Entropy Thai Sentence Segmentation with Grammar Rules](https://consensus.app/papers/details/20319672d92550ceb38e65dd5d3feee3/) — Wang et al., 2020
- [A Comparison of Thai Sentence Boundary Detection Approaches](https://consensus.app/papers/details/46247a1b177c53f99c847e415058bcc7/) — Thiengburanathum, 2020. DOI 10.1007/978-3-030-57811-4_40
- [Improving Thai Word and Sentence Segmentation Using Linguistic Knowledge](https://consensus.app/papers/details/9b80118b5d3456e7a5f9ce44f9f72818/) — Nararatwong et al., 2018. DOI 10.1587/transinf.2018edp7016
- [Syllable-based Neural Thai Word Segmentation](https://consensus.app/papers/details/9fa997ea271c59d99c23d1986f1aa86d/) — Chormai et al., 2020, COLING. DOI 10.18653/v1/2020.coling-main.407
- [Handling Cross- and Out-of-Domain Samples in Thai Word Segmentation](https://consensus.app/papers/details/6625cc0e673a554f9cc13273a2d60b3a/) — Limkonchotiwat et al., 2021, ACL Findings. DOI 10.18653/v1/2021.findings-acl.86
- [Character-based Thai Word Segmentation with Multiple Attentions](https://consensus.app/papers/details/d06bdfdd21e15b68a565769b0cc7dc12/) — Chay-intr et al., 2021
- [A comparative study on Thai word segmentation approaches](https://consensus.app/papers/details/e69217125e665d69b172ba1f5f42653a/) — Haruechaiyasak et al., 2008, 121 cites. DOI 10.1109/ecticon.2008.4600388
- [Feature-based Thai Word Segmentation](https://consensus.app/papers/details/692cd81733a75bc896d827d1d1c84369/) — Meknavin, 1997, 121 cites
- [Improving Thai Word Segmentation using HMM](https://consensus.app/papers/details/dae100bae1135b8ca36809e16fb69dda/) — Hengsanankun et al., 2020, ICSEC. DOI 10.1109/icsec51790.2020.9375142
- [Thai Word Segmentation a Lexical Semantic Approach](https://consensus.app/papers/details/232a37f1133d5f2e8ab82dadeb7e463f/) — Khankasikam et al., 2005
- [Unsupervised Word Segmentation with Bi-directional Neural LM](https://consensus.app/papers/details/e85e121faec55a0f81dce85d4884ebda/) — Wang et al., 2021, TALLIP. DOI 10.1145/3529387
- [Thai Word Segmentation with Sparse Distributed Representations](https://consensus.app/papers/details/829f7d42f2145dc185e7786fe58a475a/) — Soisoonthorn et al., 2023. DOI 10.1155/2023/8592214
- [Development of Thai word segmentation for unknown words](https://consensus.app/papers/details/243b1e3173b550fe974ca19dbcd3f585/) — Mahatthanachai et al., 2015. DOI 10.1109/icsec.2015.7401423
- [Thai Sentence Core Graph for Thai NLP](https://consensus.app/papers/details/e52c54749ff653b1b89bccd1b1563a70/) — Krukaset et al., 2024, KST. DOI 10.1109/kst61284.2024.10499681
- [A Preliminary Study on Fundamental Thai NLP Tasks for User-generated Web Content](https://consensus.app/papers/details/12e73f9fdf5d5425ae1e9ae6c0107d5f/) — Lertpiya et al., 2018. DOI 10.1109/isai-nlp.2018.8692946
- [Thai Text Processing and Its Applications](https://consensus.app/papers/details/d491901ea52053a586b6cdb36c485a99/) — Sornlertlamvanich, 2016

---

## 6. Non-paper sources

- ⭐ [Voice agent latency benchmark — how it's measured](https://openbenchmarks.com/voice-agent-latency/how-voice-agent-latency-is-measured) — the TTFAB methodology. **Read this before quoting any of our numbers against theirs.**
- [Voice agent latency comparison](https://openbenchmarks.com/voice-agent-latency/voice-agent-latency-comparison) — Vapi p50 1558 ms, Retell p50 1740 ms
- [Moshi paper PDF](https://kyutai.org/Moshi.pdf) · [code](https://github.com/kyutai-labs/moshi)
- [Typhoon2-Audio](https://github.com/scb-10x/typhoon2-audio/) — Thai speech-in/speech-out
- [Typhoon ASR Real-time announcement](https://opentyphoon.ai/blog/en/typhoon-asr-realtime-release) · [arXiv PDF](https://arxiv.org/pdf/2601.13044)
- [JaiTTS code](https://github.com/JTS-AI-Team/JaiTTS) · [RelayS2S code](https://github.com/mailong25/relays2s) · [GigaAM](https://github.com/salute-developers/gigaam) · [DuplexSLA](https://github.com/hyzhang24/DuplexSLA) · [THAI-SER](https://huggingface.co/datasets/airesearch/thai-ser) · [thonburian-tts](https://github.com/biodatlab/thonburian-tts)
- Smart Turn v3 — [announcement](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/) · [v3.2](https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/) · [model card](https://huggingface.co/pipecat-ai/smart-turn-v3) · [training code](https://github.com/pipecat-ai/smart-turn)
- [LiveKit turn-detector](https://huggingface.co/livekit/turn-detector) · [LiveKit on turn detection](https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection)
