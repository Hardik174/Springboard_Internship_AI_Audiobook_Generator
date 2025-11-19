
--- Page 1 ---
Neuro-Bloom - Learning Disability Screening & Therapy Platform

Let's take a closer look at Neuro-Bloom, a groundbreaking project developed by a team of researchers from Vishwakrama Institute of Technology in Pune, India. The team consists of Prof. Zarina Begum Mundargi, Anuj Gosavi, Avishkar Ghodke, Devang Deshpande, Jineshwari Bagul, Hardik Rokde, and others.

Abstract—Neuro-Bloom is an innovative game-based platform designed to aid in the early detection of learning disabilities in children aged 6 to 17. The system focuses on four types of learning disabilities: Dyslexia, Attention Deficit Hyperactivity Disorder (ADHD), Dyscalculia, and Dysgraphia. Through a series of interactive and engaging games, Neuro-Bloom evaluates cognitive and behavioral markers to identify potential learning problems in students.

Preliminary evaluations have shown promising accuracy rates for the system, with the Dyslexia machine learning model achieving an accuracy rate of 94%, ADHDNet at 91.7%, and varying accuracy rates for Dysgraphia between 60% and 80%.

Keywords—learning disability, ADHD, Dyslexia, Dysgraphia, Dyscalculia, behavioral patterns, engaging gameplay, specially designed games, promising accuracy, early detection.

I. Introduction

Learning disabilities affect many children worldwide, impacting their academic progress and overall development. Early identification and intervention are essential for effective management of these conditions. However, these disabilities often go undetected due to a lack of awareness, knowledge, and resources. This research aims to develop a game-based screening platform to identify children exhibiting behavioral patterns linked to common learning disabilities.

The project's goal is to bridge the gap between early signs of learning disabilities and timely professional diagnosis by providing a user-friendly

 Previously, we delved into a project that seeks to establish a game-based screening platform for early identification of children displaying behavioral patterns associated with common learning disabilities. Our primary objective is to fill the gap between the initial signs of learning disabilities and timely professional diagnosis by offering an intuitive and user-friendly solution.

The core of our project involves the development and deployment of four distinct games tailored to specific learning disabilities – dyslexia, dysgraphia, dyscalculia, and ADHD. We've designed behavioral and cognitive pattern assessments for each disorder, ensuring that each game targets key areas such as attention span, reading, number sense, and handwriting ability for every child's interaction. The games have been designed to be accessible without the need for parent or teacher supervision.

Our system is structured in a way that:
- All children in schools take our diagnostic test, which focuses on specific key features aimed at detecting early signs of learning disabilities.
- Students who score differently are shortlisted.
- Shortlisted students have one-on-one interactions with clinically verified psychiatrists, and parents are informed accordingly.
- Therapy sessions, as per the psychiatrist's verdict, take place on our platform.

Let's focus on Attention Deficit Hyperactivity Disorder (ADHD). Our platform offers three space-themed games designed for different age groups. These games help identify behavioral patterns often linked to ADHD, including inattention, impulsive behavior, and hyperactivity – traits crucial for clinical diagnosis.

Our metrics gathered from gameplay provide valuable insights to clinicians monitoring the child's behavior. For instance:
- Game 1: Galactic Defender (Wack-a-Alien) evaluates attention regulation and response consistency by having children protect a space station while ignoring visual distractions. Metrics such as Error Rate, Task Completion Time, Response Time Variability, Distracting Events, and others reflect behaviors commonly seen in clinical attention tasks.
- Game 2: Wait for Signal simulates a Go/No-Go task, measuring impulse control and visual attention. Key metrics include False Positives, Premature Responses, Gaze Shifts, Reaction Time, Playtime Duration, and more.
- Game 3: Cos

 Previously narrated excerpt:
We've previously discussed behaviors often observed in clinical attention tasks.
- Game 2: Wait for Signal mimics a Go/No-Go task, assessing impulse control and visual attention. Crucial aspects include False Positives, Premature Responses, Gaze Shifts, Reaction Time, Playtime Duration, and more.
- Game 3: Cosmic Defender tests focus and control, analyzing attention lapses, response delays, distraction handling, control errors, movement stability, input pressure, impulsive responses, reaction mistakes, and gaze shifts.

[Image 1 OCR - Tesseract]:
ADHD Diagnostic System
Game-Based Evaluation

Galactic Defender - Target Wait for Signal - Red/Green
tracking & distractions light reaction

Cosmic Pilot - Obstacle
navigation & control

This system evaluates attention lapses, response delays, distraction handling, monitors control errors, movement stability, input pressure, tracks impulsive responses, reaction mistakes, gaze shifts, and more.

Data Collection

Assessment is conducted using a Random Forest Model.

[Image 1 OCR - EasyOCR]:
ADHD Diagnostic System Game-Based Evaluation Galactic Defender Target Wait for Signal Red/Green Cosmic Pilot Obstacle tracking & distractions light reaction navigation & control Analyzes attention lapses, Monitors control errors, Tracks impulsive responses, response delays, distraction movement stability, input reaction mistakes, gaze shifts handling pressure Data Collection Evaluated using Random Forest Model Provides insight into Outputs likelihood of ADHD Model accuracy: ~88% inattention, hyperactivity, traits impulsivity

In this system, the Galactic Defender game focuses on target waiting and signal recognition, while Cosmic Pilot tests obstacle navigation and control. The system evaluates attention lapses, response delays, distraction handling, monitors control errors, movement stability, input pressure, tracks impulsive responses, reaction mistakes, gaze shifts, and more. Data is collected and analyzed using a Random Forest Model, providing insights into the likelihood of ADHD with an accuracy of approximately 88% for inattention, hyperactivity, and

 Previously narrated excerpt:
We've discussed how our tool tracks various aspects related to attention deficit hyperactivity disorder (ADHD), such as control errors, movement stability, input pressure, impulsive responses, reaction mistakes, gaze shifts, and more. Data is analyzed using a Random Forest Model, offering insights into the likelihood of ADHD with an accuracy of approximately 88% for inattention, hyperactivity, and impulsivity.

--- Page 3 ---
This game we're about to delve into is designed to assess motor regulation and precision. It features an energy bar that needs to be filled before the player can shoot at targets, requiring patience and timing - skills essential for improving impulse control and planning. The game lasts around three minutes and offers insights into cognitive and motor functions as well as behavioral control, relating to visual tracking and motor coordination tests often used in ADHD evaluations.

Metrics collected include:
- Control Failures (ft_cf): Errors in navigation or inappropriate responses indicate a lapse in attention.
- Mean Movement Variability (ft_mmv): Inconsistent movement patterns suggest difficulties with sustained attention.
- Excessive Input Intensity (ft_eii): Overuse or strong inputs might signal hyperactivity.
- Target Precision (ft_tp): Low accuracy in target alignment points to impulsivity and poor planning.

Moving on...

Dysgraphia, a learning disability affecting fine motor skills and written language production, is another area we address with our tool. Symptoms may include inconsistent handwriting, irregular spacing, spelling difficulties, and challenges in organizing ideas on paper. Overlooked during early education stages, this can lead to long-term academic challenges and decreased self-esteem.

To tackle this issue, we've developed a gamified dysgraphia screening tool featuring an interactive front-end interface powered by deep learning and generative AI. This non-invasive, child-centered initial screening can be easily implemented in school environments without requiring clinical settings or specialized oversight.

Let's delve deeper into the game design and workflow...

The game is based on a historical narrative theme, suitable for children aged 7 to 14. Upon launching the game:
- The child enters their

 Previously, we discussed the ease with which this innovative approach can be integrated into school settings without the need for clinical settings or specialized supervision.

Let's take a closer look at the game design and workflow now...

The game is centered around a historical narrative, making it suitable for children aged between 7 and 14 years old. Upon starting the game:
- The child enters their avatar.

[Image 1 OCR - Tesseract]:
v

>| >| >|
Conv2D MaxPool Flatten
Input + ReLU 2D
(256x256x3)
32 2x2 2x2
3x3 filters

pool size pool size

fT > Dysgraphia

|
yu

Dense Dense+ Dense Dense+ NonDysagaphia
+ RELU RELU +RELU Sigmoid

128 128 64 1 unit
3x filters units —_ units

Convolutional Neural Network


[Image 1 OCR - EasyOCR]:
71 Dysgraphia Conv2D MaxPool Flatten Dense Dense + Dense Dense + NonDysagaphia Input ReLU 2D RELU RELU RELU Sigmoid (256x256*3) 32 2x2 2x2 128 128 64 unit 3x3 filters pool size pool size 3x filters units units Convolutional Neural Network

This game employs a Convolutional Neural Network (CNN), which is designed to assist children with Dysgraphia. The CNN consists of several layers:
- Conv2D, MaxPool, Flatten, and Input layers, all followed by ReLU activation functions.
- Multiple Dense layers, each with RELU, RELU, RELU, and Sigmoid activation functions respectively.
- A final Dense layer that outputs a single unit, representing the child's avatar.

 Previously narrated excerpt:
The neural network structure consists of several layers, including:
- Conv2D, MaxPool, Flatten, and Input layers, all followed by ReLU activation functions.
- Multiple Dense layers, each with RELU, RELU, RELU, and Sigmoid activation functions respectively.
- A final Dense layer that outputs a single unit, representing the child's avatar.

--- Page 4 ---
2.2.3 Clinical Alignment and Feature Interpretation
The following handwriting features were identified as significant indicators of dysgraphia:

Table 1: Clinical Alignment and Feature Interpretation
The image below demonstrates the use of machine learning to identify, predict, and conclude the possibility of dysgraphia in a child, which is essential for early detection and further diagnosis.

3. Dyslexia
Dyslexia is a learning disorder that hinders a person's ability to read, write, spell, and decode words despite having the intelligence to do so. A person with this disorder experiences difficulty in processing phonemes, making it challenging to recognize and manipulate sounds in words.

Based on recent research studies on Dyslexia, significant features were selected for dyslexia prediction. As a result, we developed five games:
1. Bubble Bay: This game enhances orthographic awareness by allowing children to practice letter recognition and distinguish between similar-looking letters like n/h and b/d, which are mirror image orientations of each other. Utility: Dyslexic individuals often struggle with processing and recognizing letter forms, especially those that are mirrored or rotationally similar. This game targets a core visual perceptual difficulty associated with dyslexia.
2. Word Reef: This game helps provide lexical decision-making by requiring children to identify correctly spelled words from a collection of visually similar distractors. This helps assess visual word form recognition and error detection. Utility: Dyslexia often impairs the ability to recognize appropriate word forms and differentiate between visually confusing and phonetically similar words, thereby inferring difficulty with word-level decoding.
3. Memory Cove: This gamified experience targets working memory capacity, a cognitive domain often impaired by dyslexic

 Previously, we discussed the identification of challenges in word-level decoding due to the difficulty in recognizing appropriate word forms and differentiating between visually confusing or phonetically similar words.

Moving forward, let's delve into Memory Cove - a gamified experience designed to target working memory capacity, often affected by dyslexia.

- The Pattern Completion Game (ages 8–9) assesses logical reasoning and sequencing ability.
- The Symbol Confusion Game (ages 10–12) evaluates the child’s ability to distinguish mathematical symbols.
- The Place Value Puzzle (ages 9–11) tests comprehension of numerical structure and place value understanding.
- The Basic Word Problem Game (ages 8–10) applies mathematical reasoning to real-life situations.
- The Conversational Math Game (ages 8–12) simulates dialogue-based problem-solving to evaluate real-time math cognition.
- The Clock Reading Game (ages 9–12) assesses time interpretation skills using analog clocks.

These games are set within a candy land-themed environment to reduce test anxiety and improve engagement. Each interaction is monitored and analyzed by a Large Language Model, which evaluates not only correct answers but also the child’s reasoning process, error patterns, and consistency. This holistic approach allows for early, low-pressure detection of dyscalculia-related symptoms, ensuring the experience remains both effective and enjoyable.

Next, we'll explore the experimental results and impact of this proposed system. The system was tested with specifically designed games representing various learning scenarios. The results showcase the system’s effectiveness in detecting the behavioral indicators of learning disabilities. The value of using interactive, colorful gameplay thoughtfully designed for children of the specified target group is highlighted by these findings.

The following figures demonstrate the various parameters used in the machine learning models, the features, and the performance:

- Fig 3: Feature importance for dyslexia prediction
- Fig 4: Model Performance Comparison
- Fig 5: Impact analysis of each feature
- Fig 6: Classification Report

These graphs illustrate the parameters, features, and performance of the machine learning models used.

[Image 1 O

 Previously discussed aspects:

- Figure 3: Significance of features in predicting dyslexia
- Figure 4: Comparison of model performances
- Figure 5: Analysis of each feature's impact
- Figure 6: Classification report

These visuals offer a comprehensive view of the machine learning models' performance, parameters, and characteristics.

[Image 1 OCR - Tesseract]:
Model Performance Comparison

Decuracy      Precision      recall       F1 score
Random Forest   'Gradient Boosting

[Image 1 OCR - EasyOCR]:
Model Performance Comparison
Accuracy       Precision        Recall         F1 Score
Random Forest   Gradient Boosting

[Image 2 OCR - Tesseract]:
[Image 2 OCR - EasyOCR]:
Mode Perfonmance Comparison 1 Accutac Pecision Fl Scorc AuC Kandom forcsl Gludicnt Hoosuing Hybrd Model

[Image 3 OCR - Tesseract]:
Impact Analysis of Each Feature
fx) = 0.599

0.60   0.65   0.70   0.75
ELAX)] = 0.80    749

[Image 3 OCR - EasyOCR]:
Waterfall Plot Impact of Each Feature
fx) = 0.599
12 memory_spelling_interaction 0.08
17.929 = accuracy_to_time_ratio +0.08
40 = sentencesea_accuracy ~0.07
40 bubblebay_ accuracy ~0.06
16 phonology_orthography_interaction -0.05
691.561 total_game_time +0.04
40 = sentencesea_fallback_accuracy 0.04
37 = total hints +0.03
143.081 = sentencesea time +0.0
22 other features 0.0
0.60   0.65   0.70   0.75   0.80 E[f

 Previously Narrated Excerpt:
_ratio +0.08 (approximately 8%) for 40 sentences in the sea accuracy_
_bubblebay_ accuracy ~0.07 (around 7%) for 40 sentences_
_phonology-orthography interaction -0.05 (-5%) for 16 instances_
_total_game_time +0.04 (approximately 4%) for 691.561 minutes_
_sea_fallback_accuracy 0.04 (4%)_
_total hints +0.03 (3%) for 37 instances_
_sentencesea time +0.0 (no change) for 143.081 sentences_
_22 other features remain unchanged_
_0.60 to 0.80 on the E[f] scale_

--- Page 6 ---

Attention:
We now delve into Attention Deficit Hyperactivity Disorder (ADHD).

Fig 7: Here's a Confusion Matrix for ADHD identification.
Fig 8: This figure shows a Comparison of Evaluation Metrics for ADHD.

Section: Dysgraphia
Fig 9: The following figures display the parameters and their values.

For your listening pleasure, Figs 10 and 11 illustrate Training accuracy and validation accuracy, as well as Training loss and validation loss using epoch.

V.
CONCLUSION AND FUTURE SCOPE
A. Conclusion
Our project aims to identify learning disabilities in children early through interactive game-based screening tools. These games, designed for dyslexia, ADHD, dyscalculia, and dysgraphia, observe associated behavioral patterns and serve as primary assessment tools requiring minimal supervision. They are engaging and can be used in various schools, providing valuable insights into learning capabilities.

By detecting early warning signs without a clinical setting, we facilitate timely professional intervention. While the current implementation shows usability, further research will help to improve diagnostic accuracy and adapt the games over diverse educational and cultural contexts.

B. Future Scope
We envision several enhancements for this system's functionality and applicability:

 Previously discussed, the current implementation of our diagnostic game demonstrates usability, but further research will aid in enhancing diagnostic precision and tailoring the games to diverse educational and cultural contexts.

Moving forward, we envision several advancements for this system:

[Image 1 OCR - EasyOCR] presents a Confusion Matrix with the following statistics:
- Non-ADHD Predicted as Non-ADHD: 90
- Non-ADHD Predicted as ADHD: 70
- ADHD Predicted as Non-ADHD: 50
- ADHD Predicted as ADHD: 84

[Image 2 OCR - EasyOCR] shows the Training and Validation Loss over Epochs:
- Training Loss: 1.2
- Validation Loss: 0.8
- 1st Epoch: 1.0
- 5th Epoch: 0.6
- 14th Epoch: 0.4

[Image 3 OCR - EasyOCR] displays an Evaluation Metrics Comparison between ADHDNet and the Baseline Model:
- Accuracy: ADHDNet (91.7%), Baseline Model (84.3%)
- Precision: ADHDNet (90.5%), Baseline Model (82.2%)
- Recall: ADHDNet (93.0%), Baseline Model (85.1%)
- F1-score: ADHDNet (91.7%), Baseline Model (83.6%)
- AUC: ADHDNet (0.942), Baseline Model (0.876)

[Image 4 OCR - EasyOCR] provides the model summary for ADHDNet:
- Model Type: Sequential
- Layers: Conv2D, MaxPooling2D, Conv2D_1, MaxPooling2D_1, Conv2D_2, MaxPooling2D_2, Flatten, Dense, Dense_4, and Dense_2.
- Total Params: 11,217,297 (56.64 MB)
- Trainable Params: 

 Previously narrated excerpt:
.942), Baseline Model (0.876)

[Image 4 OCR - EasyOCR] offers a summary of ADHDNet's model:
- Type of Model: Sequential
- Components: Conv2D, MaxPooling2D, Conv2D_1, MaxPooling2D_1, Conv2D_2, MaxPooling2D_2, Flatten, Dense, Dense_4, and Dense_2.
- Total Parameters: 11,217,297 (approximately 56.64 MB)
- Trainable Parameters:

--- Page 7 ---

Let's delve into some key improvements:

1) Tailored gaming experience: Offering users a more complete and dynamic user score that is customized to their individual performance.

2) Neuro-adaptive feedback: Boosting diagnostic precision by tracking stress markers, attention transitions, and visual scanning patterns during gameplay.

3) Expansion of system infrastructure: Extending the architecture for research in neuroscience, child psychology, and linguistics, providing an in-depth analysis of the connection between gaming behavior and neural learning pathways.

By implementing these enhancements, the system can develop into a more adaptable and resilient solution, capable of meeting the increasing needs for a top-tier learning disability detection system. This could potentially aid in early identification of disabilities, ultimately leading to a better quality of life.

