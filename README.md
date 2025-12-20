# AI Public Speaking Coach

**UNC AI, IRIT-RTF, UrFU**
Telegram bot: **@BOT20007bot**

---

## 🛑 Problem Statement

Students regularly deliver presentations at:

* hackathons,
* project defenses,
* pre-defenses and thesis defenses.

The quality of these presentations directly affects:

* how the solution is perceived by the jury,
* the final grade,
* the overall impression of the speaker’s competence.

### Core Problem

Most students **do not have regular access to a professional public speaking coach**. As a result:

* there is no objective and systematic feedback;
* the same mistakes are repeated from presentation to presentation;
* communication skills improve slowly or not at all.

### Consequences

Ineffective presentations often lead to:

* loss of audience attention;
* unclear delivery of key ideas;
* misinterpretation of the solution’s value;
* lower grades and failed defenses.

In many cases, students **understand their topic well**, but fail to convincingly communicate:

* the relevance of the problem,
* the value of the solution,
* their personal contribution.

---

## 🔍 Key Aspects

To genuinely improve presentation quality, analysis must:

* rely on **objective metrics**, not subjective opinions;
* consider not only spoken text but also **paralinguistic features**, such as:

  * speech tempo,
  * pauses,
  * intonation,
  * filler words;
* provide **specific and actionable feedback**, rather than generic advice.

👉 Therefore, an automated system is needed to analyze presentation videos and generate personalized recommendations.

---

## 🎯 Project Goal

Develop an **MVP prototype** of a system that analyzes a presentation video and:

* evaluates speech and delivery,
* generates detailed feedback,
* helps users systematically improve public speaking skills.

---

## ⚙️ Core Features

### 1. 🎙 Speech-to-Text Transcription

* Convert audio to text;
* Generate timestamps to link analysis to specific moments in the presentation.

### 2. 🧠 Content Analysis

* Identify presentation structure (introduction, main body, conclusion);
* Extract key points and arguments;
* Evaluate logical flow and coherence.

### 3. 🗣 Delivery & Speech Style Analysis

* Speech tempo estimation;
* Pause and monotony analysis;
* Detection of filler words and repetitions;
* Identification of problematic time segments.

### 4. 📄 Feedback Generation

Generate a structured report including:

* strengths of the presentation;
* areas for improvement;
* concrete recommendations.

**Example feedback:**

* “Speech tempo is too fast at minute 2”
* “Use pauses after key statements to emphasize them”
* “Frequent use of the filler word ‘like’”

---

## 🔄 System Workflow

1. The user uploads a video of their presentation.
2. The system extracts the audio track.
3. Speech is transcribed into text.
4. ML models analyze:

   * textual content,
   * audio and speech characteristics.
5. The user receives a detailed feedback report with actionable advice.

---

## 📊 Data

The system uses data in the format:

**“Presentation video → Expert feedback”**

These examples are used for:

* model training,
* validation of recommendation quality,
* improving interpretability of results.

---

## 👥 Team

* **@Archjya** — ML Engineer (Speech-to-Text, audio analysis)
* **@arr3nt** — NLP Engineer (text analysis, feedback generation)
* **@pefext** — Team Lead / Backend Engineer (core pipeline)
* **@dryomahh** — Frontend / UI-UX Developer
* **@LanskihDmitry** — Product / Presentation

---

## 📦 Solution Format

The project deliverable is an **MVP system**, which will be demonstrated at the hackathon.

The presentation will cover:

* solution advantages;
* current limitations and drawbacks;
* directions for future development.

---

## 🚀 Future Development

* Analysis of non-verbal cues (gestures, posture, eye contact);
* Personalized progress tracking for users;
* Comparison with benchmark or reference presentations;
* Integration into university educational processes.

