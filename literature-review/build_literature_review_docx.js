const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, PageBreak, TableOfContents, convertInchesToTwip
} = require('docx');
const fs = require('fs');

const ACCENT = "1F3B57";
const RULE = "BFC9D4";
const HEAD_BG = "EDF1F5";

// ---------- helpers ----------
function p(text, opts = {}) {
  const runs = Array.isArray(text) ? text : [new TextRun({ text, size: opts.size || 21 })];
  return new Paragraph({
    children: runs,
    spacing: { after: opts.after === undefined ? 140 : opts.after, line: opts.line || 276 },
    alignment: opts.alignment,
    indent: opts.indent,
  });
}

// rich text: array of [text, {b,i,color}]
function rt(parts, size = 21) {
  return parts.map(([t, o = {}]) => new TextRun({
    text: t, bold: o.b, italics: o.i, color: o.color, size: o.size || size,
  }));
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 380, after: 160 },
    children: [new TextRun({ text, bold: true, size: 30, color: ACCENT })],
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 6 } },
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text, bold: true, size: 24, color: ACCENT })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 220, after: 100 },
    children: [new TextRun({ text, bold: true, italics: true, size: 22 })],
  });
}

function bullet(text, level = 0) {
  const runs = Array.isArray(text) ? text : [new TextRun({ text, size: 21 })];
  return new Paragraph({
    children: runs,
    numbering: { reference: "bullets", level },
    spacing: { after: 90, line: 268 },
  });
}

function note(parts) {
  return new Paragraph({
    children: rt(parts, 19),
    spacing: { before: 100, after: 180, line: 264 },
    indent: { left: 340 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 10 } },
  });
}

// An array cell is a list of [text, opts] parts forming one paragraph;
// a "\n" inside any part starts a new paragraph.
function partsToParagraphs(parts, size) {
  const paras = [];
  let cur = [];
  for (const [text, o = {}] of parts) {
    const chunks = String(text).split("\n");
    chunks.forEach((chunk, i) => {
      if (i > 0) { paras.push(cur); cur = []; }
      if (chunk.length) cur.push(new TextRun({ text: chunk, bold: o.b, italics: o.i, color: o.color, size: o.size || size }));
    });
  }
  paras.push(cur);
  return paras.filter(r => r.length).map(runs => new Paragraph({ children: runs, spacing: { after: 40, line: 252 } }));
}

function cell(content, { bold = false, bg, width, align } = {}) {
  const kids = Array.isArray(content)
    ? partsToParagraphs(content, 19)
    : [new Paragraph({
        children: [new TextRun({ text: content, bold, size: 19 })],
        spacing: { after: 0, line: 252 },
        alignment: align,
      })];
  return new TableCell({
    children: kids,
    width: { size: width, type: WidthType.DXA },
    shading: bg ? { type: ShadingType.CLEAR, fill: bg, color: "auto" } : undefined,
    margins: { top: 90, bottom: 90, left: 130, right: 130 },
  });
}

function table(colWidths, header, rows) {
  const trs = [];
  trs.push(new TableRow({
    tableHeader: true,
    children: header.map((t, i) => cell(t, { bold: true, bg: HEAD_BG, width: colWidths[i] })),
  }));
  for (const r of rows) {
    trs.push(new TableRow({
      children: r.map((c, i) => cell(c, { width: colWidths[i] })),
    }));
  }
  return new Table({
    columnWidths: colWidths,
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    rows: trs,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      right: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      insideVertical: { style: BorderStyle.SINGLE, size: 4, color: RULE },
    },
  });
}

function spacer(h = 120) { return new Paragraph({ children: [], spacing: { after: h } }); }

// ---------- references ----------
const refs = [
  "T. Stivers et al., \"Universals and cultural variation in turn-taking in conversation,\" PNAS, vol. 106, no. 26, pp. 10587-10592, 2009.",
  "S. C. Levinson and F. Torreira, \"Timing in turn-taking and its implications for processing models of language,\" Frontiers in Psychology, vol. 6, 731, 2015. DOI: 10.3389/fpsyg.2015.00731",
  "X. Cao et al., \"The influence of response time and feedback type on the user experience of voice interaction among older adults,\" Int. J. Human-Computer Interaction, 2024. DOI: 10.1080/10447318.2024.2348225",
  "J. Wang et al., \"Waiting experience: optimization of feedback mechanism of voice user interfaces based on time perception,\" IEEE Access, 2023. DOI: 10.1109/ACCESS.2023.3250278",
  "M. Funk et al., \"Usable and acceptable response delays of conversational agents in automotive user interfaces,\" AutomotiveUI '20, 2020. DOI: 10.1145/3409120.3410651",
  "U. Gnewuch et al., \"Faster is not always better: understanding the effect of dynamic response delays in human-chatbot interaction,\" ECIS, 2018.",
  "U. Gnewuch et al., \"Opposing effects of response time in human-chatbot interaction,\" Business & Information Systems Engineering, 2022. DOI: 10.1007/s12599-022-00755-x",
  "T. Abbas et al., \"Understanding user perceptions of response delays in crowd-powered conversational systems,\" Proc. ACM Human-Computer Interaction, 2022. DOI: 10.1145/3555765",
  "S. Asaka et al., \"Improving impressions of response delay in AI-based spoken dialogue systems,\" IEEE RO-MAN, 2024. DOI: 10.1109/RO-MAN60168.2024.10731216",
  "Openbenchmarks, \"Voice agent latency benchmark\" and \"How voice agent latency is measured,\" 2026. https://openbenchmarks.com/voice-agent-latency/",
  "Y. Qiu et al., \"Building enterprise realtime voice agents from scratch,\" arXiv:2603.05413, 2026.",
  "P. Popit et al., \"Thai-Duplex-Bench,\" JCSSE, 2026. DOI: 10.1109/JCSSE68839.2026.11596692",
  "K. Koseeyaumporn et al., \"MANJU: a multi-agent framework for natural just-in-time understanding in Thai,\" JCSSE, 2026. DOI: 10.1109/JCSSE68839.2026.11596791",
  "Kyutai Labs, \"Moshi: a speech-text foundation model for real-time dialogue,\" 2024. https://kyutai.org/Moshi.pdf",
  "Y. Liu et al., \"Discourse-aware dual-track streaming response (DDTSR),\" arXiv:2602.23266, 2026.",
  "Mai, \"RelayS2S,\" 2026.",
  "Y. Li et al., \"Style-Talker: finetuning audio language model and style-based text-to-speech model for fast spoken dialogue generation,\" 2024.",
  "S. Udupa et al., \"Endpoint anticipation for low-latency spoken dialogue,\" arXiv:2606.13450, 2026.",
  "K. Inoue et al., \"Multilingual turn-taking prediction using voice activity projection,\" 2024.",
  "E. Ekstedt and G. Skantze, \"TurnGPT: a transformer-based language model for predicting turn-taking in spoken dialog,\" Findings of EMNLP, 2020.",
  "J. P. de Ruiter, H. Mitterer and N. J. Enfield, \"Projecting the end of a speaker's turn: a cognitive cornerstone of conversation,\" Language, vol. 82, no. 3, pp. 515-535, 2006.",
  "H. Arsikere et al., \"Computationally-efficient endpointing features for natural spoken interaction with personal-assistant systems,\" IEEE ICASSP, 2014.",
  "A. Raux and M. Eskenazi, \"Optimizing endpointing thresholds using dialogue features in a spoken dialogue system,\" SIGDIAL, 2008.",
  "P. Popit et al., \"Thai semantic end-of-turn detection for real-time voice agents,\" IEEE ICSEC, 2025; arXiv:2510.04016.",
  "W. Sirichotedumrong et al., \"Typhoon ASR Real-time,\" arXiv:2601.13044, 2026.",
  "A. Andrusenko et al., \"Unified streaming and non-streaming ASR transducer training,\" 2026.",
  "V. Kutsakov et al., \"GigaAM: efficient self-supervised learner for speech recognition,\" arXiv:2506.01192, 2025.",
  "X. Li et al., \"Dynamic chunk convolution for unified streaming and non-streaming Conformer ASR,\" IEEE ICASSP, 2023. DOI: 10.1109/ICASSP49357.2023.10097062",
  "A. Kuhn et al., \"A comparison of automatic speech recognition services,\" ACM Trans. Accessible Computing, 2023.",
  "Y. Shao et al., \"Towards building speech LLMs for multitask understanding in low-resource languages,\" arXiv:2509.14804, 2025.",
  "P. Karnjanaekarin et al., \"JaiTTS: a Thai voice cloning model,\" arXiv:2604.27607, 2026.",
  "S. Thatphithakkul et al., \"The development of a Thai telephone conversational speech corpus,\" O-COCOSDA, 2023. DOI: 10.1109/O-COCOSDA60357.2023.10482925",
  "W. Aroonmanakun, \"Thoughts on word and sentence segmentation in Thai,\" 2007. DOI: 10.58837/chula.res.2007.92",
  "L. Wang et al., \"Voice-based AI in call center customer service: a natural field experiment,\" Production and Operations Management, 2023. DOI: 10.1111/poms.13953",
  "E. Brynjolfsson, D. Li and L. R. Raymond, \"Generative AI at work,\" NBER Working Paper 31161, 2023.",
  "S. V. Srinivasa Raja et al., \"A meta-analysis of the effects of AI agent implementation on customer-level outcomes,\" Journal of Business Research, 2026. DOI: 10.1016/j.jbusres.2026.116172",
  "K. Kaewtawee et al., \"Cloning a conversational voice AI agent from call recording datasets for telesales,\" arXiv:2509.04871, 2025.",
  "C. Rzepka, B. Berger and T. Hess, \"Voice assistant vs. chatbot - examining the fit between conversational agents' interaction modalities and information search tasks,\" Information Systems Frontiers, 2021. DOI: 10.1007/s10796-021-10226-5",
  "R. Dayal et al., \"Towards privacy-preserving speech intelligence: an empirical study of on-premise transcription and diarization in financial services,\" IJRASET, 2026. DOI: 10.22214/ijraset.2026.83194",
  "T. Backstrom, \"Privacy in speech technology,\" Proceedings of the IEEE, 2023. DOI: 10.1109/JPROC.2025.3632102",
  "Personal Data Protection Act B.E. 2562 (2019), Kingdom of Thailand, ss. 27-29 (cross-border transfer). Summary: DLA Piper, Data Protection Laws of the World - Thailand.",
  "Gowajee, \"Voice Agent Platform,\" https://gowajee.ai/ (accessed 27 Aug 2026).",
  "GPTBots, \"GPTBots partners with Thai voice tech leader Tellvoice,\" press release, 29 Sep 2025.",
  "Helport AI, \"New Bangkok location to strengthen Helport AI's multilingual AI service delivery network,\" Nov 2025.",
  "S. Kim et al., \"Temporal focus of self-supervised models for tone,\" arXiv:2511.12285, 2025.",
  "D. Isbell et al., \"Cross-linguistic forced alignment for Thai,\" JASA, 2025.",
  "S. Ok et al., \"SpeculativeETD: end-of-turn detection with a tiered local/server model,\" 2025.",
  "Z. Wang et al., \"FastTurn: unifying acoustic and streaming semantic cues for turn-taking,\" arXiv:2604.01897, 2026.",
];

const refParas = refs.map((r, i) => new Paragraph({
  children: [new TextRun({ text: `[${i + 1}]  `, bold: true, size: 18 }), new TextRun({ text: r, size: 18 })],
  spacing: { after: 80, line: 250 },
  indent: { left: 460, hanging: 460 },
}));

// ---------- document body ----------
const body = [];

// Title block
body.push(new Paragraph({
  children: [new TextRun({ text: "A Low-Latency Thai Voice-Agent Loop", bold: true, size: 44, color: ACCENT })],
  spacing: { before: 900, after: 80 },
  alignment: AlignmentType.CENTER,
}));
body.push(new Paragraph({
  children: [new TextRun({ text: "Literature Review, Applications, and Motivation", size: 28, color: "44586E" })],
  spacing: { after: 260 },
  alignment: AlignmentType.CENTER,
}));
body.push(new Paragraph({
  children: [new TextRun({ text: "Speech-to-Text -> Large Language Model -> Text-to-Speech, in Thai, under 500 ms", italics: true, size: 21, color: "5A6B7C" })],
  spacing: { after: 500 },
  alignment: AlignmentType.CENTER,
}));
body.push(new Paragraph({
  children: [new TextRun({ text: "Chanachon Ruanpheng", size: 22 })],
  spacing: { after: 40 }, alignment: AlignmentType.CENTER,
}));
body.push(new Paragraph({
  children: [new TextRun({ text: "27 August 2026", size: 20, color: "5A6B7C" })],
  spacing: { after: 640 }, alignment: AlignmentType.CENTER,
}));

body.push(new Paragraph({
  children: [new TextRun({ text: "Scope of this report", bold: true, size: 22, color: ACCENT })],
  spacing: { after: 100 },
}));
body.push(p([...rt([
  ["This report answers two questions put to the project after the first demonstration of the speech-to-speech loop: ", {}],
  ["(1) what does the existing literature say about building this kind of system, and (2) who would actually use it and why.", { b: true }],
  [" Sections 2 to 5 answer the first; Sections 6 and 7 answer the second. Section 8 states what has been measured on our own hardware, Section 9 states the gaps the literature leaves open, and Section 10 states the limitations honestly, including the ones that weaken our own claims.", {}],
])]));
body.push(note([
  ["A note on evidence. ", { b: true }],
  ["Every number in this report is labelled with where it came from. Figures marked ", {}],
  ["measured here", { b: true }],
  [" were produced on our own hardware and can be reproduced from the repository. Everything else is a published or vendor claim and is treated as a lead, not a fact. Section 4 explains why this distinction matters more in this field than in most: published voice-agent latencies are not measured from the same instant and mostly cannot be compared to each other.", {}],
]));

body.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 1. Introduction
body.push(h1("1.  Introduction and problem statement"));
body.push(p("A voice agent is a program that a person talks to over the phone or a microphone and that talks back. The dominant way to build one is a cascade of three models: automatic speech recognition (ASR) turns the caller's audio into text, a large language model (LLM) decides what to say, and text-to-speech (TTS) turns that reply back into audio. Everything in this report concerns the loop that connects those three, in Thai, on hardware we control."));
body.push(p([...rt([
  ["The engineering problem is not accuracy. It is ", {}],
  ["time", { b: true }],
  [". A caller stops speaking and then waits. The interval between the true end of their speech and the first sound of the agent's reply is the quantity this project exists to minimise. Our target is a median of under 500 ms. Section 2 explains where that number comes from, and Section 3 explains what is known about whether it actually matters to the person waiting.", {}],
])]));
body.push(p("Two things make Thai a distinct problem rather than an instance of a solved one. First, the components that make English loops fast, in particular streaming ASR that emits partial transcripts while the caller is still talking, do not exist for Thai as released checkpoints. Second, the models that decide when a caller has finished speaking are trained on other languages and, as Section 4.3 shows, do not transfer."));

// ---- 2. Where the latency target comes from
body.push(h1("2.  Where the latency target comes from"));
body.push(p("The 500 ms target is not an arbitrary engineering ambition. It is derived from two findings in the psycholinguistics of conversation."));
body.push(p([...rt([
  ["Stivers et al. [1] measured turn transitions across ten languages sampled worldwide, from small indigenous communities to major world languages. Every language they examined avoided overlap and minimised silence between turns, and the between-language variation in mean gap fell ", {}],
  ["within 250 ms of the cross-language mean", { b: true }],
  [", clustering near 200 ms. This is the empirical basis for treating roughly 200 ms as the natural gap in human conversation.", {}],
])]));
body.push(note([
  ["Caveat we have not closed. ", { b: true }],
  ["The abstract of [1] does not name its ten languages, so whether Thai or any other Tai-Kadai language is in the sample is unverified. The universality claim makes a 200 ms design target defensible for Thai by extension, not by direct measurement. Obtaining the paper's language table is an open item.", {}],
]));
body.push(p([...rt([
  ["Levinson and Torreira [2] then state the puzzle that gives this project its architecture. Human turn gaps run around 200 ms, but the latency to ", {}],
  ["produce", { i: true }],
  [" language is over 600 ms. Both cannot be true of a system that waits for evidence that the other person has stopped and only then begins working. The conclusion they draw is that humans ", {}],
  ["predict", { b: true }],
  [" the end of the current turn and prepare their response before it arrives.", {}],
])]));
body.push(note([
  ["Why this is the single most important citation in the report. ", { b: true }],
  ["Our system detects the end of a turn reactively, and so does every commercial product we could find. If [2] is right, that entire category has a latency floor it cannot cross, and our largest stage is not badly tuned but structurally misconceived. Section 4.2 covers the published work that acts on this.", {}],
]));

// ---- 3. Does latency matter
body.push(h1("3.  Does response latency actually matter to users?"));
body.push(p("A latency target is only worth pursuing if delay damages the interaction. The human-computer interaction literature broadly supports this for voice, but with an important qualification that argues against optimising blindly."));

body.push(h2("3.1  Evidence that delay hurts"));
body.push(bullet(rt([
  ["Cao et al. [3] measured older adults using a smart speaker and found the ", {}],
  ["optimal dialogue response time to be 600 to 1500 ms", { b: true }],
  [", with satisfaction, comfort and perceived anthropomorphism all decreasing as response time grew, and a significant drop beyond 3000 ms.", {}],
])));
body.push(bullet(rt([
  ["Wang et al. [4] studied voice user interface feedback timing directly and identified ", {}],
  ["750 ms as the optimal feedback point", { b: true }],
  [" and ", {}],
  ["1850 ms as the threshold beyond which user emotion turns negative", { b: true }],
  [" on both arousal and valence.", {}],
])));
body.push(bullet(rt([
  ["Funk et al. [5] found that an unnaturally long delay leads users to ", {}],
  ["assume an error has occurred", { b: true }],
  [", which changes their behaviour rather than merely their mood.", {}],
])));
body.push(bullet(rt([
  ["Abbas et al. [8], at much larger delays, found the waiting experience evaluated negatively beyond 8 seconds, and that higher-quality responses made users underestimate elapsed time.", {}],
])));
body.push(note([
  ["Read against our own numbers. ", { b: true }],
  ["Our current measured median is about 1375 ms (Section 8). That sits above the 750 ms optimum of [4] and inside the 600 to 1500 ms band of [3], but below the 1850 ms point where [4] reports emotional cost. The literature therefore says our present system is tolerable and not yet good, which is a fair description of it.", {}],
]));

body.push(h2("3.2  The counter-evidence, which we do not hide"));
body.push(p([...rt([
  ["Gnewuch et al. [6] found the opposite effect in text chat: dynamically ", {}],
  ["delayed", { i: true }],
  [" chatbot responses increased perceived humanness, social presence and overall satisfaction relative to near-instant ones. Their later work [7] reconciles this by showing the effect depends on the user's prior experience with chatbots, and that response time acts as a social cue rather than purely as a cost. Tan et al. (2026) similarly found that participants rated LLM output produced after a 2-second delay as ", {}],
  ["less", { i: true }],
  [" thoughtful than the same class of output after 9 or 20 seconds.", {}],
])]));
body.push(p([...rt([
  ["This matters for the honesty of the project's motivation. The correct claim is not \"faster is always better\". It is narrower and still sufficient: in ", {}],
  ["spoken, task-oriented", { b: true }],
  [" interaction, where the caller is holding a phone and waiting for an answer, delay is a cost [3][4][5]. The humanness benefit reported in [6] was found in ", {}],
  ["text", { b: true }],
  [" chat, where a delay is legible as typing. In a phone call, silence is legible as a dropped line.", {}],
])]));
body.push(p([...rt([
  ["A practical corollary appears in Asaka et al. [9], who mitigated unavoidable delay in a cascaded ASR-LLM-TTS system with ", {}],
  ["filler insertion", { b: true }],
  [" (a 1.1-point Likert improvement in the impression of delay) and parallel speech processing (0.6 s of real reduction). This is worth noting because it is a cheap win our loop does not yet implement.", {}],
])]));

// ---- 4. Literature review, technical
body.push(h1("4.  Literature review: how the field builds these systems"));

body.push(h2("4.1  Three architectural families"));
body.push(p("Every voice-to-voice system we found falls into one of three families, and the field moved toward the third during 2026."));
body.push(spacer(80));
body.push(table([2000, 4100, 2900],
  ["Family", "How it works", "Reported latency"],
  [
    [[["Cascaded", { b: true }], ["\nASR -> LLM -> TTS", {}]],
     "Three models in sequence, overlapped where possible. Paralinguistic information is discarded at the ASR boundary.",
     [["600 ms budgeted; 1500-1700 ms measured on real calls [10]", {}], ["\nUsed by Vapi, Retell, LiveKit, Pipecat, and this project.", { i: true }]]],
    [[["End-to-end speech-to-speech", { b: true }]],
     "One model, audio in and audio out, often full-duplex. Keeps prosody, but reported to be semantically weaker.",
     [["Moshi ~200 ms practical [14]; GPT-Realtime-Mini 1.5 s and Gemini Live 4.2 s in Thai [12]", {}]]],
    [[["Hybrid / speculative", { b: true }]],
     "Run a fast path and a slow path together, commit the fast one, let the better one refine behind it.",
     [["Speech-to-speech onset with cascade quality; RelayS2S [16] reports P90 onset comparable to pure S2S at 99% of cascaded quality", {}]]],
  ]));
body.push(spacer(140));
body.push(p([...rt([
  ["Two findings settle the architecture question for Thai in favour of the cascade, at least for now. Qiu et al. [11], surveying enterprise voice agents, state plainly that ", {}],
  ["\"fully self-hosted end-to-end solutions are not yet available\"", { b: true }],
  [". And Thai-Duplex-Bench [12], discussed in Section 4.4, found that a ", {}],
  ["cascaded reference baseline achieved the highest strict pass rate, 97.0%", { b: true }],
  [", on Thai customer-service role adherence, beating both commercial end-to-end systems it tested.", {}],
])]));

body.push(h2("4.2  Endpointing: the field says predict, not detect"));
body.push(p("Endpointing is the decision of when the caller has finished speaking. In our system it is the single largest stage, 642 ms of a 1375 ms loop, or 47% (Section 8). It is therefore where the literature matters most."));
body.push(p([...rt([
  ["Following [2], the most relevant recent work forecasts turn ends rather than detecting them. Udupa et al. [18] forecast end-of-turn signals ", {}],
  ["up to 2.56 s in advance", { b: true }],
  [", enabling speculative execution of the LLM and TTS on partial context, and report a ", {}],
  ["505 ms average latency reduction for 28.4% more computation", { b: true }],
  [". Critically for us, the method is speech-based and needs no streaming ASR, which is the component Thai does not have.", {}],
])]));
body.push(p("A second family reaches similar ends far more cheaply. Arsikere et al. [22] discriminate non-final from final pauses using only inexpensive acoustic features (periodicity, speaking rate, spectral constancy, duration and intensity, pitch of pre-pausal speech), with no speech recognition, no speaker information and no session information, reaching 20% equal error rate at 100 ms of latency. Their own baseline is directly comparable to ours: a fixed 500 ms threshold produced a 36% cutoff rate, against our 600 ms window's 18% (Section 8). Raux and Eskenazi [23] earlier showed that setting the endpointing threshold dynamically from dialogue features cuts latency by up to 24% versus a fixed threshold."));
body.push(p("On the text side, TurnGPT [20] is the canonical model: a transformer predicting turn shifts from text alone, whose attention analysis shows it genuinely uses dialogue context and pragmatic completeness rather than surface cues. De Ruiter et al. [21], in a much-cited psycholinguistic study, provide the theoretical warrant for the text route: lexicosyntactic content is necessary and possibly sufficient for projecting turn completion, whereas intonation is neither. Flattening pitch entirely left turn-end projection accuracy unchanged."));

body.push(h2("4.3  Why models trained on other languages fail here"));
body.push(p([...rt([
  ["Inoue et al. [19] tested Voice Activity Projection models across languages and found that a ", {}],
  ["monolingual turn-taking model does not transfer to another language", { b: true }],
  [", while a multilingual model trained on English, Mandarin and Japanese matched monolingual performance on all three and learned to discern the input language.", {}],
])]));
body.push(note([
  ["This is the published explanation for a failure we measured. ", { b: true }],
  ["We tested Smart Turn v3.2, a widely used audio turn-detector advertised for 23 languages, on Thai zero-shot. It scored 1 of 4 on our chronic cases and was ", {}],
  ["confidently", { i: true }],
  [" wrong, assigning probabilities of 0.873 to 0.979 to speakers who then kept talking for another 1.6 to 2.2 seconds. Thai is absent from its language list. The lesson from [19] is that this is not because Thai is tonal, since Mandarin is tonal and is handled; it is that these models do not transfer at all, and the remedy is multilingual training rather than a better monolingual model.", {}],
]));
body.push(p("Thailand-specific work does exist. Popit et al. [24] fine-tuned a 0.6B classifier for Thai end-of-turn detection, reporting F1 0.881 at about 110 ms. No weights were released, so the result is a target rather than a component."));

body.push(h2("4.4  Streaming ASR for low-resource languages"));
body.push(p("The cascade cannot overlap work with the caller's speech unless the recogniser emits partial transcripts as the caller talks. For Thai, no released checkpoint does this. Typhoon ASR Real-time [25] is the strongest Thai recogniser we tested and is described as streaming, but the released configuration is full-context, so it emits nothing until the utterance ends. Pathumma and Thonburian are offline. This is the single largest missing component in the Thai stack."));
body.push(p("The literature does supply recipes. Andrusenko et al. [26] train one transducer for both offline and streaming operation using chunk-limited attention with right context, dynamic chunked convolutions and mode-consistency regularisation, and open-source the framework. GigaAM [27] samples chunk sizes dynamically during pretraining so one model can be fine-tuned into either mode, and is MIT-licensed. Both are directly applicable to a Thai fine-tune."));
body.push(note([
  ["The cost nobody quotes. ", { b: true }],
  ["Kuhn et al. [29] found that across eleven commercial services, streaming mode measures significantly worse accuracy than batch. Li et al. [28] price the degradation and the remedy: naive streaming costs 41.7% relative, dynamic chunk convolution recovers most of it down to 16.7%. Any Thai streaming fine-tune must therefore measure the accuracy it spends, not assume it is free.", {}],
]));

body.push(h2("4.5  The Thai-specific literature"));
body.push(p("The Thai voice-agent literature is small enough to enumerate. The following are the papers that bear directly on this project."));
body.push(spacer(80));
body.push(table([2500, 4200, 2300],
  ["Work", "What it contributes", "Status for us"],
  [
    ["Thai-Duplex-Bench [12]", "555 items, 1015 clips, two tracks including a service track across five customer-service domains. Baselines: GPT-Realtime-Mini 1.5 s median time-to-audio, Gemini Live 4.2 s; cascaded baseline highest strict pass at 97.0%.", "Independent Thai support for our cascaded architecture"],
    ["MANJU [13]", "The only published Thai voice-to-voice loop we could find: 1.8 s to first audio, 6.07 s end to end.", "The Thai baseline our numbers should be read against"],
    ["Thai semantic end-of-turn [24]", "Fine-tuned 0.6B classifier, F1 0.881 at ~110 ms, trained from subtitle boundaries.", "Target; no weights released"],
    ["Typhoon ASR Real-time [25]", "Thai FastConformer transducer. Shows rigorous text normalisation matches the impact of model scaling; releases a human-labelled Thai ASR benchmark.", "In use (accuracy); rejected as streaming"],
    ["XLSR-Thai [30]", "First Thai self-supervised speech encoder, 36,000 hours, plus Thai-SUP (>1,000 h spoken language understanding).", "Untested; the encoder for a Thai turn model"],
    ["JaiTTS v1.0 [31]", "Thai TTS adapted from VoxCPM, the same base we run. CER 1.94% against human ground truth's 1.98%; 283 wins to 58 losses in 400 pairwise human comparisons.", "Untested; a direct A/B candidate"],
    ["Thai telephone corpus [32]", "37h28m of real 8 kHz agent-to-customer calls, 4,383 customers. Its five most frequent words are final particles.", "The corpus this problem needs; availability unconfirmed"],
    ["Thai segmentation [33]", "Argues Thai discourse should be seen as a combination of clauses rather than sentences; space is not a reliable boundary marker.", "Governs how we split replies for streaming TTS"],
  ]));

// ---- 5. The measurement problem
body.push(h1("5.  A finding about the literature itself: the numbers are not comparable"));
body.push(p("The most consequential result of this review is methodological. Published voice-agent latencies are measured from different instants, and most of them cannot be compared to one another."));
body.push(spacer(80));
body.push(table([1900, 4300, 2800],
  ["Figure", "What it actually measures", "Source"],
  [
    ["~250 ms", "A component-sum budget: ASR 60-100, LLM time-to-first-token 100-180, TTS first chunk 40-80, WebRTC 20-40. Nothing is measured end to end.", "Vendor architecture guides"],
    ["~600 ms", "A well-tuned pipeline sum, end of user speech to first audio out. Still a sum, not an observation.", "Vendor blogs"],
    [[["1558 ms / 1740 ms", { b: true }]], "Median time-to-first-audio-byte on real phone calls: caller stops speaking to agent audio begins, read off the saved recording. Vapi over 382 turns; Retell.", "Openbenchmarks 2026 [10]"],
    ["1.5 s / 4.2 s", "Median time-to-audio in Thai for GPT-Realtime-Mini and Gemini Live.", "Thai-Duplex-Bench [12]"],
  ]));
body.push(spacer(140));
body.push(p([...rt([
  ["The budget figures and the measured figures differ by ", {}],
  ["2.5 to 3 times for the same products", { b: true }],
  [". The benchmark's own methodology page concedes that no independently verified study measures across every latency layer, and that available tests measure different sections of the call path with different models and different levels of disclosure [10].", {}],
])]));
body.push(p([...rt([
  ["This has a direct consequence for our own work. Our clock originates at ", {}],
  ["hand-marked true end-of-speech", { b: true }],
  [", not at the moment our voice-activity detector fires. That is the same origin the Openbenchmarks suite uses, which makes those the first external numbers in this project's history that are legitimately comparable to ours. It also means a lazier endpointer cannot flatter our metric, which is why the origin was chosen that way.", {}],
])]));

// ---- 6. Use cases
body.push(h1("6.  Use cases: who would use this, and for what"));
body.push(p("The system is a loop, not an application. What it enables is any product where a Thai speaker talks to software over a voice channel and expects an answer at conversational speed. The following are the applications with evidence behind them, ordered by how well that evidence holds."));
body.push(spacer(80));
body.push(table([2400, 3500, 3100],
  ["Application", "What the agent does", "Why low latency and Thai quality matter here"],
  [
    [[["Inbound contact centre", { b: true }], ["\n(primary)", { i: true }]],
     "Answers customer calls: balance enquiries, order status, policy questions, routing to a human when needed.",
     "The caller is on a phone with nothing to look at. Silence reads as a dropped call [5]. Recognition failures measurably increase demand for a human agent and increase complaints [34]."],
    [[["Outbound operations", { b: true }]],
     "Payment reminders, appointment confirmation, customer verification, loan pre-qualification.",
     "High volume and repetitive, so unit cost dominates; these are the workloads Thai vendors already name [42]."],
    [[["Telesales and qualification", { b: true }]],
     "Follows a structured playbook learned from top-performing human agents.",
     "Studied in Thai by Kaewtawee et al. [37]: blind tests show the agent approaching human performance on routine parts of the call while underperforming on persuasion and objection handling."],
    [[["Voice front-end to internal systems", { b: true }]],
     "Hands-busy or eyes-busy staff querying stock, schedules, or records by speaking.",
     "Rzepka et al. [38] found speech shows higher perceived efficiency, lower cognitive effort and higher satisfaction than text for information search."],
    [[["Regulated-sector intake", { b: true }], ["\n(healthcare, finance)", { i: true }]],
     "First-line triage and data capture where the recording contains personal or financial detail.",
     "The audio cannot leave the organisation, which forces self-hosting [39][40]. This is where our architecture is not merely competitive but required."],
  ]));

body.push(h2("6.1  What the evidence says about voice agents in call centres"));
body.push(p([...rt([
  ["The strongest single study is Wang et al. [34], a natural field experiment in a large telecommunications company that replaced an interactive voice response system with a voice-based AI system. The introduction of the AI system ", {}],
  ["persistently reduced customer complaints", { b: true }],
  [", with the largest effects for simple requests and for older and female customers. Brynjolfsson et al. [35], studying 5,172 support agents, found AI assistance raised issues resolved per hour by 15% on average, with the largest gains for less experienced workers.", {}],
])]));
body.push(p([...rt([
  ["The same field experiment supplies the finding that most directly justifies this project's technical emphasis: ", {}],
  ["speech-recognition failures in customer-AI interactions led to increases in customers' demand for human service and in customer complaints", { b: true }],
  [" [34]. In other words, the failure mode that destroys the business case for a voice agent is the ASR layer, and for Thai callers the ASR layer is precisely what generic multilingual systems get wrong (Section 7.1).", {}],
])]));
body.push(note([
  ["The counterweight, stated plainly. ", { b: true }],
  ["A 2026 meta-analysis of 468 effect sizes across 95 articles and 82,751 participants [36] finds that, on average, customers respond ", {}],
  ["less", { i: true }],
  [" favourably to AI agent implementation, attributing the aversion to a prejudicial bias triggered when AI intrudes into domains previously handled by humans. The direction and magnitude depend on design, information and context. The honest reading is that a voice agent is not automatically welcome, and that quality of interaction, of which latency and recognition accuracy are components, is what decides whether it is tolerated.", {}],
]));

body.push(h2("6.2  The Thai deployment landscape"));
body.push(p("Thai-language voice agents are already being sold, which is evidence of demand rather than a reason to stop."));
body.push(bullet(rt([
  ["Gowajee", { b: true }],
  [" [42], a Chulalongkorn University spin-off founded in 2024, sells a Thai-first voice agent platform with speech-to-text and text-to-speech \"built natively for Thai, not translated from English\", a no-code flow editor, and contact-centre automation. Named customers include Generali Life Assurance (Thailand), TV Direct and Forth Smart Service. Its published material states no accuracy figure, no latency figure, and no deployment model.", {}],
])));
body.push(bullet(rt([
  ["Tellvoice", { b: true }],
  [", a long-established Thai speech technology company, partnered with GPTBots in September 2025 to bring Thai voice recognition into that platform, targeting banking enquiries and voice biometric authentication [43].", {}],
])));
body.push(bullet(rt([
  ["Helport AI", { b: true }],
  [" opened a Bangkok office in November 2025 focused on Thai-language operations for financial institutions and lending platforms [44].", {}],
])));
body.push(note([
  ["Market sizing was attempted and is not reported. ", { b: true }],
  ["Published figures for the Thai contact-centre market found in this review (for example USD 49.7 billion in 2024, or a call-centre platform market of USD 1,028 billion) come from press-release aggregators and are implausible against a national GDP of roughly USD 515 billion. They are excluded rather than cited. A defensible market estimate would require primary sources (National Statistical Office employment data, or operator disclosures) and is listed as open work in Section 9.", {}],
]));

// ---- 7. Why ours
body.push(h1("7.  Why anyone would use this loop rather than an existing one"));
body.push(p("Three arguments, in decreasing order of how well we can evidence them."));

body.push(h2("7.1  Thai last-mile quality, which generic systems do not have"));
body.push(p([...rt([
  ["Measured here: on our frozen 22-utterance Thai corpus, Typhoon ASR reaches a character error rate of ", {}],
  ["0.032", { b: true }],
  [" against Whisper-large's ", {}],
  ["0.130", { b: true }],
  [", a factor of four. We also observed a failure mode specific to multilingual models on low-resource input: Whisper emitted Katakana, Vietnamese and Russian ", {}],
  ["inside a single Thai sentence", { b: true }],
  [", at a CER of 0.762, with the language explicitly pinned to Thai.", {}],
])]));
body.push(p("Thai-Duplex-Bench [12] makes the same observation about the field as a whole, motivating its own existence with the remark that several systems claim multilingual support including Thai, yet no open benchmark had specifically measured Thai conversational behaviour. Combined with [34]'s finding that recognition failures drive callers to demand humans, the Thai last-mile is not a refinement; it is the thing that decides whether the deployment survives."));

body.push(h2("7.2  Latency, measured the same way as the commercial systems"));
body.push(p("Because our clock shares an origin with the Openbenchmarks methodology (Section 5), the following comparison is legitimate, subject to the three caveats immediately below it."));
body.push(spacer(80));
body.push(table([3200, 1800, 4000],
  ["System", "Median", "Conditions"],
  [
    ["Human conversation [1]", "~200 ms", "Ten languages, natural conversation"],
    ["Moshi (English) [14]", "~200 ms", "End-to-end speech model, L4 GPU, no telephony"],
    ["GPT-Realtime-Mini, Thai [12]", "1500 ms", "Commercial end-to-end, Thai-Duplex-Bench"],
    [[["This project", { b: true }], ["\ntyphoon_llamacpp", { i: true }]], [["1375 ms", { b: true }]], [["Measured here", { b: true }], [": 22-item frozen corpus, quiet box, no telephony leg", {}]]],
    ["Vapi [10]", "1558 ms", "382 real phone turns, including telephony"],
    ["Retell AI [10]", "1740 ms", "Real phone calls, including telephony"],
    ["MANJU, Thai [13]", "1800 ms", "The only published Thai voice-to-voice loop"],
    ["Gemini Live, Thai [12]", "4200 ms", "Commercial end-to-end, Thai-Duplex-Bench"],
  ]));
body.push(spacer(140));
body.push(note([
  ["Three caveats, any one of which could eat the gap. ", { b: true }],
  ["(1) The commercial numbers include a telephony leg that ours excludes; our loop legitimately stops at the wire because customers bring their own telephony, but a customer's end-to-end experience would add it back. (2) Ours is a laboratory number on a 22-item corpus; theirs is 382 live phone turns with network jitter and interruptions. (3) Our test machine drifts 25 to 45% in throughput across an evening, so 1375 ms is relative to its measurement window. The direction of the comparison is real; the precision is not.", {}],
]));
body.push(p([...rt([
  ["Stated conservatively, the claim is this: ", {}],
  ["the shipped state of the art on real calls is 1.5 to 1.7 s, Thai-specific commercial systems run 1.5 to 4.2 s, and the only published Thai loop is 1.8 s to first audio. Reaching 500 ms would be roughly three times better than anything currently sold.", { b: true }],
])]));

body.push(h2("7.3  Self-hosting, which is a compliance requirement before it is a preference"));
body.push(p("Everything in the loop runs on one machine with no cloud hop inside the latency path. That decision was made for speed, but it independently answers a regulatory problem."));
body.push(p([...rt([
  ["Thailand's Personal Data Protection Act restricts transfers of personal data outside the country under sections 28 and 29, requiring an adequacy finding or appropriate safeguards such as standard contractual clauses or binding corporate rules [41]. A call recording is personal data twice over: it contains what the caller said, and it contains their voice, which is a biometric identifier [40]. Dayal et al. [39] make the equivalent argument for regulated finance, concluding that fully on-premise ASR and diarization are practical with open-source components and motivated by privacy and sovereignty regulation. Bäckström [40] catalogues the wider risk: speech carries side information about health, emotion, affiliation and relationships, all of it private.", {}],
])]));
body.push(p("Qiu et al. [11] observe that fully self-hosted end-to-end solutions are not yet available. That is the gap this project sits in: a Thai voice loop that a bank or hospital can run inside its own network is not currently something they can buy."));

// ---- 8. What we have measured
body.push(h1("8.  What has been measured on our own hardware"));
body.push(p("This section states our own results so the advisor can see what is claim and what is evidence. All figures were produced on a single RTX 5060 Ti (16 GB) with speech-to-text, LLM and text-to-speech co-resident, against a frozen 22-utterance Thai corpus, timed from hand-marked true end of speech."));
body.push(spacer(80));
body.push(table([2600, 1700, 4700],
  ["Stage", "Median (ms)", "Note"],
  [
    [[["Endpointing", { b: true }]], [["642", { b: true }]], [["47% of the loop and by far the largest stage. It did not grow; everything around it shrank.", { b: true }]]],
    ["Speech-to-text flush", "88", "Typhoon ASR on the CPU. Was 541 ms with Whisper."],
    ["LLM first token", "124", "Was 453 ms; see the ollama finding below."],
    ["Clause generation to first cut", "219", "100% decode time; the splitter never actually cuts on this corpus."],
    ["Text-to-speech first byte", "225", "VoxCPM. Alone on an idle card it reaches 110-120 ms."],
    [[["Total, end of speech to first audio", { b: true }]], [["1375", { b: true }]], [["Best configuration; paired against the previous best twice each on a machine verified quiet before each run.", {}]]],
  ]));
body.push(spacer(140));
body.push(p("Four results are worth reporting to an examiner because they are negative or surprising."));
body.push(bullet(rt([
  ["The inherited endpointing window already cuts callers off. ", { b: true }],
  ["At a 600 ms silence window, 4 of 22 corpus utterances are truncated by 1.6 to 4.5 seconds, and those four are cut at every window worth having because their interior pauses are longer than any usable threshold. This was invisible before instrumenting it, because the clock silently re-originated to the early trigger.", {}],
])));
body.push(bullet(rt([
  ["Thai mid-turn and turn-final pauses are barely separable by duration. ", { b: true }],
  ["Measured on the LOTUSDIS Thai conversational corpus (8 sessions, about 2 hours, 1,137 same-speaker resumptions against 396 floor changes): median pause 288 ms when the speaker resumes, 416 ms when the floor changes. The best duration-only threshold is 325 ms at a ", {}],
  ["41.9% equal error rate", { b: true }],
  [". Published acoustic-feature work reaches 20% EER [22], so roughly half the error is recoverable from signal that a pause-length timer throws away.", {}],
])));
body.push(bullet(rt([
  ["The Thai final-particle heuristic has a measured ceiling of 23.3%. ", { b: true }],
  ["On 4,688 real Thai floor changes, only 1,093 end with a polite final particle (khrap, kha and similar), against 8.7% of mid-turn pauses. The signal is real and far too weak to endpoint on: a particle rule misses three-quarters of genuine turn ends.", {}],
])));
body.push(bullet(rt([
  ["An inference server was charging 360 ms of pure overhead. ", { b: true }],
  ["One identical request cost 471 ms through ollama and 108 ms sent straight to the llama-server subprocess ollama itself launches. Removing that layer moved the loop's median from 1639 ms to 1375 ms. Decode throughput was unchanged at about 52 tokens per second, which is why the effect appeared entirely in first-token latency.", {}],
])));

// ---- 9. Gaps
body.push(h1("9.  What the literature does not answer"));
body.push(p("These are the open questions that make the project a research contribution rather than an integration exercise. Each is stated as something we could not find rather than something we claim to be first at."));
body.push(bullet(rt([
  ["No streaming Thai ASR checkpoint exists. ", { b: true }],
  ["The best Thai recogniser is packaged as streaming and, by its own configuration file, is not. Producing one is a limited-context fine-tune following [26] or [27], not a download.", {}],
])));
body.push(bullet(rt([
  ["No Voice Activity Projection model covers Thai, ", { b: true }],
  ["and Thai appears in no VAP training set we found. Given [19], this is why zero-shot turn-detection fails here.", {}],
])));
body.push(bullet(rt([
  ["No end-to-end latency figure measured from true end of speech exists for any self-hosted Thai stack. ", { b: true }],
  ["Ours may be the only one, which is a publishable observation in itself provided the caveats in Section 7.2 travel with it.", {}],
])));
body.push(bullet(rt([
  ["Vendor budgets and measured reality are unreconciled. ", { b: true }],
  ["A 2.5 to 3 times gap between advertised and observed latency for the same products [10] is documented but not explained anywhere we found.", {}],
])));
body.push(bullet(rt([
  ["No published pause-duration distribution for Thai conversation. ", { b: true }],
  ["The numbers in Section 8 appear to be the first, and they set the floor for what any timer-based Thai endpointer can achieve.", {}],
])));
body.push(bullet(rt([
  ["No defensible Thai market sizing. ", { b: true }],
  ["As noted in Section 6.2, the available figures are not usable. This is open work.", {}],
])));

// ---- 10. Limitations
body.push(h1("10.  Limitations and ethical position"));
body.push(bullet(rt([
  ["The measurement conditions are favourable to us. ", { b: true }],
  ["A 22-item frozen corpus on a quiet single-user machine with no telephony, no network jitter, no barge-in and one concurrent call. Every number in Section 8 should be read with that attached.", {}],
])));
body.push(bullet(rt([
  ["The test machine drifts. ", { b: true }],
  ["The same configuration has read 25 to 45% apart across a single evening, and not equally between software stacks, so cross-session comparisons cannot be repaired by scaling. Only figures bracketed back-to-back in one measurement window are quoted as comparisons.", {}],
])));
body.push(bullet(rt([
  ["The evaluation corpus is scraped and cannot judge turn-taking. ", { b: true }],
  ["Its clips end mid-thought and only 3 of 22 close with a final particle. Evaluating endpointing properly needs real two-party Thai conversation, either a recording session or the telephone corpus of [32].", {}],
])));
body.push(bullet(rt([
  ["The current voice is a research artefact and is not shippable. ", { b: true }],
  ["The text-to-speech voice was cloned from public video without the speaker's consent. That is acceptable for private research and is not acceptable the moment the system is demonstrated publicly or takes money. Replacing it is a recording session with a consenting voice actor, not a retrain, because the model clones from a reference clip. Everything stays private until then. This is recorded here deliberately rather than left for someone to discover.", {}],
])));
body.push(bullet(rt([
  ["Most of the literature in this report has not been reproduced. ", { b: true }],
  ["Of the works surveyed, a small number have been tested on our hardware; the rest are indexed with an explicit status of untested. Two of the most relevant papers are recent preprints with no citations. They are leads.", {}],
])));

// ---- 11. Conclusion
body.push(h1("11.  Conclusion"));
body.push(p([...rt([
  ["The literature supports the project on three points and warns it on two. It supports the latency target, which derives from measured human turn-taking [1][2] and is corroborated by voice-interface studies placing user tolerance between roughly 750 and 1850 ms [3][4]. It supports the cascaded architecture for Thai specifically, since a cascaded baseline beat both commercial end-to-end systems on Thai service-role adherence [12]. And it supports the use case, since voice AI in call centres measurably reduces complaints [34] while recognition failures measurably increase them [34], which is exactly the axis a Thai-specialised system competes on.", {}],
])]));
body.push(p([...rt([
  ["It warns the project twice. First, reactive endpointing may be the wrong category rather than a badly tuned stage [2], which means the 642 ms that dominates our loop is unlikely to yield to tuning and should be attacked by anticipation [18] or by cheap acoustic features [22]. Second, customers do not welcome AI agents by default [36], so latency and Thai accuracy are necessary conditions for the product to be tolerated, not sufficient conditions for it to be wanted.", {}],
])]));
body.push(p([...rt([
  ["The immediate next step follows from Section 8 rather than from any paper: endpointing is 47% of the loop, its Thai pause statistics are now measured for the first time, and the gap between our 41.9% equal error rate on duration alone and the 20% reported from cheap acoustic features [22] is the largest identified and unclaimed improvement in the system.", {}],
])]));

// ---- References
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(h1("References"));
body.push(...refParas);

// ---------- build ----------
const doc = new Document({
  creator: "Chanachon Ruanpheng",
  title: "A Low-Latency Thai Voice-Agent Loop: Literature Review, Applications, and Motivation",
  description: "Literature review and use-case analysis for a Thai speech-to-speech voice agent loop",
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21 }, paragraph: { spacing: { line: 276 } } },
      heading1: { run: { font: "Calibri", color: ACCENT } },
      heading2: { run: { font: "Calibri", color: ACCENT } },
      heading3: { run: { font: "Calibri" } },
    },
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 460, hanging: 260 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 860, hanging: 260 } } } },
      ],
    }],
  },
  sections: [{
    properties: {
      page: { margin: { top: 1300, right: 1300, bottom: 1300, left: 1300 } },
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(process.argv[2], buf);
  console.log("written:", process.argv[2], buf.length, "bytes");
});
