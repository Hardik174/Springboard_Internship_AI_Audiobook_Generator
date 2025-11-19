import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
import pdfplumber
import PyPDF2
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Utility Functions
# ------------------------------

def wer(ref, hyp):
    """Word Error Rate (S + D + I) / N"""
    ref_words = ref.split()
    hyp_words = hyp.split()
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words)
    S = D = I = 0
    for opcode, i1, i2, j1, j2 in sm.get_opcodes():
        if opcode == 'replace':
            S += max(i2 - i1, j2 - j1)
        elif opcode == 'delete':
            D += (i2 - i1)
        elif opcode == 'insert':
            I += (j2 - j1)
    N = len(ref_words)
    return (S + D + I) / N if N > 0 else 0

def cer(ref, hyp):
    """Character Error Rate (S + D + I) / N"""
    sm = difflib.SequenceMatcher(None, ref, hyp)
    S = D = I = 0
    for opcode, i1, i2, j1, j2 in sm.get_opcodes():
        if opcode == 'replace':
            S += max(i2 - i1, j2 - j1)
        elif opcode == 'delete':
            D += (i2 - i1)
        elif opcode == 'insert':
            I += (j2 - j1)
    N = len(ref)
    return (S + D + I) / N if N > 0 else 0

def text_similarity(ref, hyp):
    """Cosine similarity between reference and hypothesis text"""
    vectorizer = TfidfVectorizer().fit([ref, hyp])
    tfidf = vectorizer.transform([ref, hyp])
    return min(1.000000000000000, cosine_similarity(tfidf[0], tfidf[1])[0][0])

# ------------------------------
# Extract text using 4 libraries
# ------------------------------

def extract_text_pymupdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def extract_text_pdfminer(pdf_path):
    return pdfminer_extract_text(pdf_path).strip()

def extract_text_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_pypdf2(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

# ------------------------------
# Main Evaluation
# ------------------------------

def evaluate(pdf_path, reference_text):
    results = {}

    libs = {
        "PyMuPDF": extract_text_pymupdf,
        "pdfminer.six": extract_text_pdfminer,
        "pdfplumber": extract_text_pdfplumber,
        "PyPDF2": extract_text_pypdf2
    }

    for name, extractor in libs.items():
        try:
            extracted = extractor(pdf_path)
            results[name] = {
                "WER": wer(reference_text, extracted),
                "CER": cer(reference_text, extracted),
                "Text Similarity": text_similarity(reference_text, extracted)
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


# ------------------------------
# Example usage
# ------------------------------

if __name__ == "__main__":
    pdf_path = "C:\\Users\\Hardik Rokde\\Downloads\\Group7_SNA.pdf"

    # Paste reference text from your PDF here
    reference_text = """Social Media Integration System for Hoax Call
 Campaign Detection
 Devam Desai∗, Nitu Sherawat†, Samar Borkar‡
 Department of Information Technology
 Indian Institute of Information Technology Allahabad, India
 Email: ∗iit2022035@iiita.ac.in, †iit2022073@iiita.ac.in, ‡iit2022083@iiita.ac.in
 Abstract—
 I. OBJECTIVE
 Problem: The increasing misuse of social media platforms
 for coordinating hoax call campaigns poses a significant chal
lenge to public safety and law enforcement. These campaigns
 can spread false information rapidly and cause harm, leading
 to public panic.
 Goal: The objective of this project is to develop a stan
dalone system that can extract, analyze, and integrate social
 media data related to hoax call campaigns. The system aims
 to detect potential hoax communications, map user networks
 to uncover coordination, correlate social media activity with
 actual hoax events, and assess the potential threat posed by
 these campaigns.
 II. DATASETS USED
 • Social Media Data: Data was collected from multiple
 platforms including Reddit and Wikipedia focusing on
 posts that could potentially be related to hoax call cam
paigns.
 • Sample Dataset: The dataset contains anonymized social
 media posts with fields such as content, user ID, times
tamp, post ID, and labels indicating whether the content
 is hoax-related.
 III. DATA COLLECTION
 • Reddit: Using the PRAW (Python Reddit API Wrapper)
 and Pushshift.io API to collect posts from relevant sub
reddits.
 • Wikipedia: Using the Wikipedia-Scraper to gather data
 from public Facebook posts.
 • Database Link
 IV. LITERATURE REVIEW
 A. Introduction
 Social media platforms have become integral to contem
porary online communication and the dissemination of in
formation. Among these platforms, Reddit stands out due
 to its unique structure, which revolves around user-created
 communities known as subreddits, each focused on specific
 topics.1 This organization fosters diverse discussions and the
 formation of specialized communities, making Reddit a rich
 source of data for studying social interactions, the spread of
 information, and the evolution of online collectives. However,
 the open and dynamic nature of such platforms also presents
 critical challenges, notably the proliferation of misinformation,
 including fake news and hoaxes, alongside the constant need
 to understand the nuances of user behavior and the veracity
 of shared content.5 Robust analytical techniques are there
fore essential to navigate this complex landscape, enabling
 researchers and users alike to discern credible information and
 understand the dynamics of online discourse.
 This literature review aims to synthesize the key method
ologies, findings, and datasets from a selection of influential
 research papers that address social network analysis within the
 Reddit ecosystem, various approaches to fake news detection
 applicable to social media data, and sentiment analysis tech
niques relevant to understanding user opinions. By examining
 these interconnected areas, this review seeks to provide a
 comprehensive foundation for understanding the current state
 of research and the challenges and opportunities associated
 with analyzing social interactions and information veracity on
 platforms like Reddit.
 The evolution of platforms such as Reddit reveals a complex
 ecosystem where understanding user interactions and content
 reliability is of utmost importance. The initial scarcity of
 subreddits on Reddit, which expanded over time 1, indicates
 a platform that developed organically around the evolving
 interests of its users. This progression from a limited number
 of general topics to a vast and diverse array of specialized
 subreddits reflects the increasing fragmentation and special
ization of online communities. This evolution has significant
 consequences for how information spreads, the formation of
 communities, and the nature of interactions within them. Fur
thermore, the interconnectedness of social network analysis,
 fake news detection, and sentiment analysis becomes apparent
 when considering the necessity of understanding both the
 structural aspects of online interactions and the characteristics
 of the content being shared. For example, the sentiment
 expressed within a co-comment network could provide insights
 into the level of agreement or polarization surrounding specific
 topics, potentially influencing the dissemination and impact
 of fake news. These three areas are not independent; the
 structure of a social network can impact how fake news
 spreads, and the sentiment conveyed in the content can be a
crucial feature in identifying misinformation. A comprehensive
 analysis of social media requires an understanding of these
 interdependencies.
 B. Co-Comment Network Construction
 The study ”Co-Comment Network: A Novel Approach
 to Construct Social Networks within Reddit” proposes an
 innovative method for automatically building social networks
 within Reddit by analyzing user co-commenting behavior.19
 This approach addresses the fact that while Reddit hosts a
 multitude of discussions, it lacks explicit social networking
 features such as friend connections or direct messaging.20 The
 researchers posit that users who comment on the same posts
 likely share mutual interests, and this shared interest can be
 leveraged to construct a social network.19
 The methodology involves several key steps. First, for every
 pair of users, the number of posts on which both users have
 commented is identified. Second, the similarity between each
 pair of users is calculated using Jaccard’s coefficient, which
 measures the ratio of the number of common posts to the total
 number of unique posts commented on by either user. Third,
 the weight of the connection between two users is determined
 by multiplying their similarity score with the total number of
 posts commented on by both users combined. This weighting
 scheme accounts for both the degree of shared interest and
 the overall activity of the users within the network. Finally,
 by applying this weight calculation to every user pair in the
 Reddit dataset, a weight matrix is constructed, representing the
 Co-Comment network where users are nodes and the weights
 of the undirected edges between them indicate the strength
 of their connection based on shared commenting activity.
 Figure 2 in 19 provides a visual overview of this project idea,
 and Figure 3 illustrates how connections between users are
 established based on their co-commenting activity.
 A central focus of this study is to examine whether
 these constructed ”Co-Comment” networks exhibit properties
 characteristic of real-world social networks.19 The analysis
 confirms that the Co-Comment network conforms to these
 desirable social network properties, such as a power-law distri
bution in node degrees, indicating the presence of a few highly
 active users and many with fewer connections. The network
 also demonstrates the small-world effect, characterized by a
 low average shortest path between any two connected users,
 and a significant clustering coefficient, suggesting that users
 connected to a common user are also likely to be connected
 to each other.19 Furthermore, the researchers applied the
 Louvain method, a heuristic algorithm based on modularity
 optimization, to detect communities within these Co-Comment
 networks.19 These communities represent groups of users with
 a higher density of connections among themselves compared
 to the rest of the network. To validate the constructed network,
 a semantic evaluation was performed on the discovered com
munities by analyzing the distribution of subreddit categories
 within them. This analysis revealed that users within specific
 communities tend to comment on posts related to similar
 topics, indicating a coherence and shared interest within these
 groups, such as communities focused on electronic cigarettes
 or sports.
 The co-comment network approach effectively demonstrates
 that social networks can emerge organically from user in
teractions on platforms not explicitly designed for social
 networking. The tendency of users to comment on specific
 types of posts 19 suggests a form of self-organization around
 shared interests. This implies that user behavior can reveal
 underlying social structures even in the absence of explicit
 social ties. Understanding these implicit networks can provide
 valuable insights into community formation and information
 f
 low. The confirmation that Co-Comment networks exhibit
 properties like the small-world effect and power-law distri
bution 19 reinforces the idea that online platforms can mirror
 real-world social dynamics. These properties are indicative of
 efficient information spread and the presence of influential
 nodes within the network. Identifying these properties helps
 in understanding how information and influence propagate
 within Reddit communities. The use of the Louvain method
 to identify communities within the Co-Comment network 19
 and the semantic validation of these communities highlight the
 potential for automatically discovering topical interest groups
 within Reddit. This information can be valuable for targeted
 content recommendation, understanding opinion polarization,
 or identifying potential areas for misinformation spread, as
 communities represent groups of users with stronger internal
 connections and shared interests.
 C. Community Detection Enhancement
 The paper ”Hybrid Louvain-Clustering Model Using
 Knowledge Graph for Improvement of Clustering User’s Be
havior on Social Networks” explores a method to enhance
 community detection in social networks by integrating a
 knowledge graph with the Louvain clustering algorithm.25
 Traditional social network analysis faces limitations in effec
tively clustering content based on the diverse and complex
 nature of user behavior.29 To address these limitations, the
 authors propose a hybrid model that leverages the structured
 information provided by a knowledge graph to improve the
 clustering of user behaviors on social networks like Face
book.25
 The proposed model operates in five key steps.25 The first
 step involves the automatic collection of data from Facebook
 using the Scrapy tool. The second step focuses on trans
forming the collected Facebook data into a knowledge graph.
 This involves detecting various entities such as Users, Pages,
 Groups, Statuses, Images, Albums, Comments, and Reactions,
 and representing their relationships as a directed graph where
 nodes are entities and edges represent the connections between
 them, with associated weights. The third step, detailed in
 Algorithm 1 of the paper, calculates the weights of the directed
 relationships from users to pages based on user interactions
 with content. This involves identifying the page associated
 with a user’s reaction to a status or image and assigning a
 weight based on the type and number of interactions. The
 fourth step calculates the weights of relationships between
D. Temporal Dynamics
 pages by considering the total number of interaction rela
tionships of users’ behavior divided by the total number of
 users, resulting in weighted relationships among pages based
 on shared user interactions. The fifth and final step involves
 embedding the weights calculated in the previous steps as
 input variable values for the Louvain algorithm.
 The Louvain algorithm is a modularity-based method widely
 used for detecting communities in large networks.25 It has
 been applied in various online social networks, including
 Reddit, Twitter, and YouTube, to cluster users based on their
 behavior or to group topics based on the co-occurrence of
 terms.25 By incorporating the weights derived from the knowl
edge graph, the hybrid model enhances the clustering process
 compared to the traditional Louvain algorithm, which typically
 treats all connections as equal.25 Experimental results con
ducted on a large-scale Facebook social network, comprising
 120,000 nodes from January to October 2019, demonstrated
 that the proposed hybrid approach significantly outperforms
 the traditional Louvain method in clustering content based on
 user behavior.25 The hybrid model achieved better precision
 and recall in clustering contents based on users’ behavior,
 indicating its effectiveness in capturing the nuances of user
 interactions.
 The integration of knowledge graphs with traditional clus
tering algorithms like Louvain offers a powerful approach to
 capturing the complexity of user behavior in social networks.
 By representing relationships and attributes in a structured
 format, knowledge graphs can provide richer information for
 community detection than simple network topologies. The
 improved performance demonstrated by the hybrid model 25
 suggests that considering the nature and intensity of user
 interactions leads to more meaningful clusters. Traditional
 clustering might treat all interactions as equal, whereas the
 hybrid model leverages the semantic information within the
 knowledge graph to differentiate between types and strengths
 of relationships, allowing for a more accurate representation
 of user affinities and interests. The application of the Louvain
 algorithm in various social network contexts 25 highlights its
 versatility and effectiveness as a modularity-based community
 detection method for large graphs. However, the need for
 hybrid approaches indicates that the traditional Louvain al
gorithm may have limitations in capturing the nuances of user
 behavior, particularly in real-time and dynamic environments.
 The potential for inconsistencies in Louvain results across
 different implementations 31 also suggests the importance of
 careful implementation and parameter tuning. The focus on
 clustering content based on user behavior 25 has significant
 implications for applications such as recommendation systems
 and targeted advertising. By accurately identifying groups
 of users with similar interests and behaviors, platforms can
 deliver more relevant content and improve user engagement.
 The proposed model’s ability to handle real-time data 25 is
 particularly valuable in dynamic social network environments,
 as understanding how users interact with content is key to
 personalizing their experience on social media.
 The study ”Understanding the Evolution of Reddit in Tem
poral Networks induced by User Activity” investigates the
 platform’s evolution by constructing and analyzing temporal
 networks based on user activity across different subreddits
 over time.1 The research aims to understand the dynamics
 of evolving fields of common interest by analyzing these
 temporal networks using community detection methods.2
 The methodology for building these temporal networks
 involves analyzing the Pushift dataset from 2006 to 2019 on
 a year-over-year basis.2 The approach developed by Huber is
 used to create weighted edges between pairs of subreddits.2
 This is achieved by normalizing the number of users active
 in two subreddits with the number of users active in the
 respective neighborhood of each individual subreddit.2 The
 resulting edge weight serves as a measure of distance or the
 strength of interest between subreddits.1 The induced network
 is considered a map where user activity with shared interests
 connects subreddits into communities.1 Due to the large
 number of users, the initial networks are fully connected.1
 To address this, an edge reduction step is introduced where,
 for each year, the edge weight that breaks the connected
 component is determined, and edges below this threshold are
 removed to make community detection more effective.1
 The Leiden Algorithm, a modularity-based community de
tection method, is then applied to these temporal networks to
 examine the number of communities, the size of the largest
 community, and the center of the largest community over
 time.1 The analysis reveals several key findings. The number
 of communities shows a steep increase between 2007 and
 2011, peaking in 2008 with 108 individual communities,
 before stabilizing at approximately five communities between
 2011 and 2019.1 This suggests that the evolution of sub
reddit communities can be explained by the coalescence of
 decentralized and independently developing structures.1 The
 size of the largest community shows an upward trend over
 time, supporting the hypothesis that individual communities
 grow by incorporating smaller ones, indicating the emergence
 of influential topics or subreddits.1 However, analysis of the
 center of the largest community (the most central subreddit)
 for each year reveals a constant alternation, with no single sub
reddit remaining the most central across consecutive years.1
 This suggests that while there are rapidly growing subreddits
 driving the evolution, there isn’t just one consistently domi
nant subreddit of importance.1 Preliminary experiments with
 10,000 subreddits indicate that rapidly growing subreddits are
 indeed a key factor in the temporal evolution of Reddit user
 activity.1
 The temporal analysis of Reddit reveals a platform that has
 undergone significant structural changes over time, moving
 from a more fragmented landscape of interests to a more
 consolidated one with a few dominant communities. The
 initial increase in the number of communities followed by a
 decline and stabilization 1 suggests a period of exploration
 and diversification followed by a convergence around core
topics. This evolution could be attributed to various factors,
 including platform growth, the introduction of new features
 (e.g., user-created subreddits as mentioned in 4), and shifts
 in user interests over time. Understanding these temporal
 dynamics is crucial for predicting future trends and the po
tential impact of events on the platform. The observation
 that the center of the largest community (the most important
 subreddit) changes frequently 1 indicates a dynamic landscape
 of user interests within the dominant community. While a
 large community might exist, the specific topics or subreddits
 driving its core can shift from year to year. This suggests
 that even within seemingly stable community structures, there
 is ongoing evolution and change, highlighting the need for
 continuous monitoring and analysis of online communities
 to understand the shifting priorities and interests of their
 members. The preliminary finding that rapidly growing sub
reddits are key drivers of Reddit’s evolution 1 points to the
 importance of understanding the factors that contribute to the
 rapid growth of online communities. Identifying and analyzing
 these drivers could provide insights into emerging trends, viral
 content, and the potential for new topics or communities to
 gain prominence on the platform. This also connects to the
 concept of ”Eternal September” 35 where a large influx of
 new users can impact community dynamics.
 V. FAKE NEWS DETECTION TECHNIQUES
 A. r/Fakeddit Dataset
 The paper ”r/Fakeddit: A New Multimodal Benchmark
 Dataset for Fine-grained Fake News Detection” introduces a
 novel, large-scale, multimodal dataset designed to advance
 research in the detection of fake news.5 Recognizing the
 limitations of existing fake news datasets, which often suf
fer from limited size, single modality (typically text), and
 coarse-grained categorization, the authors present r/Fakeddit
 as a comprehensive resource for developing more robust and
 nuanced detection models.6
 The dataset is notable for its scale, comprising over 1
 million samples 5, significantly larger than many prior datasets
 in the field. A key characteristic of r/Fakeddit is its multi
modality.5 Each sample includes text in the form of Reddit
 submission titles, images as thumbnails associated with the
 submissions, metadata such as the submission score, author’s
 username, subreddit source, domain, number of comments,
 and up-vote to down-vote ratio, and user comments made
 on the submissions.5 This multimodal nature addresses a
 significant gap in previous datasets that often focused solely
 on textual content. Furthermore, r/Fakeddit offers fine-grained
 fake news categorization.5 Samples are labeled according
 to 2-way (fake/true), 3-way (completely true/fake with true
 text/fake with false text), and 6-way classification schemes.5
 The 6-way labels provide a more detailed classification, in
cluding categories such as True, Satire/Parody, Misleading
 Content, Imposter Content, False Connection, and Manipu
lated Content. The data is sourced from 22 different subreddits
 on Reddit, spanning a significant temporal range from March
 2008 to October 2019, ensuring a diverse collection of topics
 and perspectives.6 The dataset underwent several stages of
 review to ensure quality and relevance to subreddit themes.6
 Beyond basic fake news classification, r/Fakeddit holds po
tential for various other applications, such as implicit fact
checking, where one modality (e.g., an image) can be used to
 verify another (e.g., the accompanying text).6 The dataset is
 publicly available on GitHub 5, and an associated challenge
 aims to benchmark progress in developing models capable of
 accurately detecting specific types of fake news in text and
 images.5
 The r/Fakeddit dataset represents a crucial advancement in
 fake news research by providing a large-scale, multimodal
 resource with detailed labels. The inclusion of diverse data
 types like images, metadata, and comments alongside text 5
 enables researchers to develop more sophisticated models that
 can consider multiple forms of evidence when assessing the
 authenticity of news, mirroring the multifaceted way humans
 often evaluate information. The fine-grained categorization
 of fake news 5 is particularly valuable as it acknowledges
 the heterogeneous nature of misinformation, allowing for the
 creation of models that can distinguish between different types
 of fake news, each potentially requiring unique detection
 strategies. The sourcing of the dataset from Reddit 6 offers
 a specific context for studying fake news within a particular
 type of online community, allowing for insights relevant to this
 platform and potentially generalizable to others with similar
 characteristics. The temporal span of the data 6 also provides
 an opportunity to study the evolution of fake news trends over
 time.
 B. Machine Learning Approaches
 1) SGD Classifier:
 • Research has explored the application of the Stochastic
 Gradient Descent (SGD) classifier for the task of distin
guishing between real and fake news articles.9 The SGD
 algorithm is a computationally efficient method for train
ing linear classifiers and regression models, particularly
 useful for large datasets as it updates model parameters
 using small batches of data at a time.9 It is also capable
 of online learning, adapting to new data as it becomes
 available.9
 • In the context of fake news detection, the SGD classifier
 is often used with text-based features extracted from
 news articles.9 Common feature extraction techniques
 include Term Frequency-Inverse Document Frequency
 (TF-IDF) and Count Vectorizer, which convert text data
 into numerical representations that can be processed
 by the algorithm.9 Several studies have reported high
 accuracy using the SGD classifier for fake news detection.
 For instance, one study achieved an accuracy of 94%
 with test datasets using an SGD classifier with TF-IDF
 features.48 Another study comparing various machine
 learning algorithms for fake news detection found that
 SGD achieved an accuracy of 96.11% on a publicly
 available fake news dataset.51 The performance of SGD
 has also been compared with other machine learning
algorithms such as Random Forest, Logistic Regression,
 and Support Vector Machines in some research, providing
 insights into its relative effectiveness.46 One paper even
 proposed a model using multimodal fusion to identify
 fake news, achieving high accuracy on the Gossipcop and
 Fakeddit datasets.41
 • The successful application of the SGD classifier in multi
ple studies 48 indicates its effectiveness as a computation
ally efficient algorithm for fake news detection, especially
 in scenarios involving large datasets or the need for online
 learning. Its performance, often reaching high accuracy
 levels, suggests that linear models can be surprisingly
 effective when combined with appropriate feature engi
neering techniques like TF-IDF. The comparison of SGD
 with other machine learning algorithms 46 highlights the
 importance of evaluating different models and feature sets
 for optimal performance in fake news detection. While
 SGD can be effective, other algorithms like Support Vec
tor Machines, Logistic Regression, and ensemble methods
 might outperform it in certain contexts or with different
 datasets. This underscores the need for experimentation
 and careful selection of the most appropriate model for a
 given task and dataset. The focus on text-based features
 9 in many of these studies indicates the importance of
 linguistic analysis in fake news detection. Techniques
 like TF-IDF and more advanced word embeddings 46
 can capture crucial differences in the language used in
 fake and real news articles. However, with the increasing
 prevalence of multimodal fake news, relying solely on
 text might have limitations, emphasizing the need to
 incorporate other modalities like images and metadata,
 as highlighted by the r/Fakeddit dataset.41
 2) Comparative Analysis: The paper ”Hoax News Clas
sification using Machine Learning Algorithms” provides a
 comparative analysis of several machine learning algorithms
 for classifying news as hoax or not.10 The study evaluates the
 performance of Support Vector Machine (SVM), Na¨ ıve Bayes,
 Decision Tree, Logistic Regression, Stochastic Gradient De
scent (SGD), and Neural Network (MLP) based on metrics
 such as accuracy, precision, recall, and F-measure.55
 The findings indicate that the Neural Network (MLP) algo
rithm generally achieved the highest performance, with an av
erage of 93% for accuracy, F-measure, and precision.55 How
ever, the Support Vector Machine (SVM) algorithm yielded
 the highest recall value of 94%.55 Logistic Regression also
 demonstrated strong performance with a precision of 93% and
 an accuracy of 91%. The Decision Tree algorithm showed
 a balanced performance across metrics, with an accuracy of
 92%. Stochastic Gradient Descent (SGD) achieved an accuracy
 of 89% in this study, while Na¨ ıve Bayes had a lower precision
 of 85% but a high recall of 92%, resulting in an accuracy of
 88%. These results are consistent with findings from other
 studies that compare various machine learning algorithms for
 fake news detection. For example, research has shown that
 Passive-Aggressive Classifiers, Random Forests, and Support
 Vector Machines are efficient methods for automatic false
 news detection.49 Deep learning models like BERT have also
 demonstrated significant enhancements in fake news detection
 compared to traditional feature extraction techniques.50 The
 performance of these algorithms heavily relies on feature
 engineering and vectorization techniques such as TF-IDF, Bag
 of Words, and word embeddings like BERT, which play a
 crucial role in representing the textual data in a format suitable
 for machine learning models.11
 The comparative studies reveal that different machine learn
ing algorithms exhibit varying strengths and weaknesses in
 detecting hoax and fake news. While Neural Networks (MLP)
 often demonstrate high overall performance 55, SVM might
 excel in identifying all actual fake news (high recall), and
 simpler models like Logistic Regression can also achieve
 strong precision. This suggests that the optimal algorithm
 choice depends on the specific objectives of the detection
 system. The consistently strong performance of algorithms
 like Neural Networks, SVM, and Logistic Regression across
 multiple studies indicates their robustness for fake news detec
tion. However, the variability in reported performance across
 different datasets and studies 50 highlights the importance
 of evaluating models on diverse and representative data, as
 no single algorithm is universally superior. The increasing
 use of deep learning models like BERT and LSTM 50 in
 recent research suggests a trend towards leveraging more
 advanced techniques that can capture contextual information
 and subtle linguistic cues more effectively than traditional
 machine learning algorithms. The high accuracy reported for
 models like BERT 50 indicates the potential of these advanced
 techniques for significantly improving fake news detection.
 3) User Interaction Signals (Some Like it Hoax): The paper
 ”Some Like it Hoax: Automated Fake News Detection in
 Social Networks” proposes an innovative approach to fake
 news detection by focusing on the users who ”liked” Facebook
 posts, rather than analyzing the content of the posts them
selves.63 The study posits that patterns in user interaction can
 provide strong signals about the veracity of information.
 The dataset used in this research comprises 15,500 Face
book posts and 909,236 users.63 The posts originate from
 Facebook pages known to share either scientific content or
 conspiracy theories and fake scientific news.65 The researchers
 assume that posts from scientific pages are reliable (non
hoaxes), while posts from conspiracy pages are hoaxes.65
 Two primary classification techniques are proposed and eval
uated.63 The first is Logistic Regression, where the set of
 users who liked a post are used as features to train a model to
 predict whether the post is a hoax or not. The second technique
 is Harmonic Boolean Label Crowdsourcing, an adaptation of
 crowdsourcing algorithms that iteratively updates scores for
 users (representing their propensity to like hoaxes or non
hoaxes) and posts (representing the probability of being a
 hoax) based on the pattern of ”likes” and a small set of labeled
 training data.
 Both techniques achieved remarkably high classification ac
curacies, exceeding 99%, even when the training set contained
 less than 1% of the posts.63 Furthermore, the techniques
 proved robust, maintaining high accuracy even when consid
ering only users who liked both hoax and non-hoax posts.63
 The code developed for this research is publicly available.66
 The ”Some Like it Hoax” paper demonstrates that user
 engagement patterns, particularly who interacts with content,
 can be a powerful indicator of its veracity. The extremely high
 accuracies achieved using user ”likes” as features 63 suggest
 that communities of users may exhibit distinct preferences for
 interacting with certain types of information. The success of
 both logistic regression and the adapted harmonic boolean
 label crowdsourcing algorithm 63 highlights the potential
 of different machine learning and statistical techniques for
 leveraging user interaction data. The harmonic algorithm’s
 ability to propagate information across users who liked the
 same posts addresses a limitation of logistic regression. This
 approach also raises important questions about the dynamics of
 online communities and the potential for echo chambers, as the
 tendency for hoaxes and non-hoaxes to be liked by different
 user groups suggests a level of polarization in information
 consumption.
 VI. SENTIMENT ANALYSIS
 Tool: VADER (Valence Aware Dictionary and sEntiment
 Reasoner)
 • Lexicon-based sentiment analyzer.
 The Valence Aware Dictionary and sEntiment Reasoner
 (VADER) is a lexicon and rule-based sentiment analysis
 tool specifically designed to perform well on social media
 text.12 Unlike many other sentiment analysis tools that
 require training on large datasets, VADER relies on a
 built-in lexicon containing over 7,500 words, phrases, and
 emoticons, each rated for its sentiment intensity (valence)
 on a scale from-4 (extremely negative) to +4 (extremely
 positive).12 This lexicon was empirically validated by
 human judges and is particularly attuned to the language
 used in social media, including slang, acronyms, and
 emoticons.74
 In addition to its lexicon, VADER employs a set of
 heuristic rules to account for contextual nuances that
 can affect the sentiment expressed in text.12 These rules
 consider the impact of punctuation, such as exclamation
 points and question marks, which can amplify senti
ment.14 Capitalization is also taken into account, with
 words in all caps often indicating increased sentiment
 intensity.14 Degree modifiers, like ”very” or ”extremely,”
 are recognized for their ability to intensify or de-intensify
 the sentiment of adjacent words.14 Furthermore, VADER
 handles negations by considering the presence of words
 like ”not” that can flip the sentiment of a word.14
 The tool also considers the sentiment shifting effects of
 conjunctions.15
 VADER computes a compound sentiment score for a
 given text, which is a normalized, weighted compos
ite score ranging from-1 (most negative) to +1 (most
 positive).12 This score provides an overall measure of
 the sentiment expressed in the text. VADER also returns
 separate scores for the proportion of positive, negative,
 and neutral sentiment in the text.12 One of the key
 advantages of VADER is its ease of use, as it does not
 require any training data.12 It is also relatively fast and
 has been shown to be effective in analyzing the sentiment
 of social media text.12 However, like any tool, VADER
 has limitations, and it may struggle with understanding
 more complex linguistic phenomena like sarcasm and
 irony.12
 VADER’s design, specifically tailored for social media
 text, makes it a valuable tool for analyzing sentiment in
 online discussions, including those on Reddit. Its ability
 to quickly and easily provide sentiment scores without the
 need for extensive training data allows for rapid analysis
 of large volumes of user-generated content. The com
pound sentiment score offers a convenient single metric
 for gauging the overall sentiment, while the separate
 positive, negative, and neutral scores provide a more
 nuanced understanding. The applications of VADER in
 areas like brand monitoring, market research, and political
 analysis highlight its broad utility in understanding public
 opinion in online environments. Within the context of
 Reddit, VADER could be used to analyze sentiment
 towards specific subreddits, topics, or user interactions,
 providing valuable insights for researchers studying on
line communities and information dynamics.
 VII. METHODOLOGY
 A. Edge Extraction
 • Content Analysis: The text of the posts is preprocessed
 using NLTK and SpaCy for tokenization, lemmatization,
 and removal of stop words. Entities like keywords, hash
tags, and mentions are extracted from the posts to identify
 potential signals of coordinated hoax activities.
 • Edge Creation: User interactions (e.g., replies, mentions,
 shares) were treated as edges to build a user interaction
 network, representing the potential connections between
 individuals involved in hoax campaigns.
 B. Classification
 • SGD Classifier: For content analysis, a Stochastic
 Gradient Descent (SGD) classifier was used to classify
 social media posts as either hoax-related or not. This
 classifier was chosen due to its efficiency and ability to
 handle large datasets.
 TF-IDF (Term Frequency-Inverse Document Frequency)
 was used as the feature extraction technique for convert
ing the text data into numerical representations that the
 SGD classifier can process.
 The classifier was trained on labeled data (posts marked
 as hoax-related or not), and its performance was evaluated
 using accuracy, precision, recall, and F1-score.
C. Sentiment Analysis
 VIII. SYSTEM ARCHITECTURE
 • VADER Sentiment Analysis: The VADER (Valence
 Aware Dictionary and sEntiment Reasoner) sentiment
 analysis technique was used to assess the sentiment of the
 social media posts. VADER is particularly effective for
 social media content as it takes into account the nuances
 of informal language, such as slang and emoticons.
 The sentiment scores produced by VADER were used
 to classify posts into positive, neutral, or negative senti
ments. Additionally, the negative sentiment scores were
 correlated with the potential threat levels of hoax calls.
 D. Temporal Analysis
 • Temporal Correlation: Temporal analysis was con
ducted to examine the relationship between the times
tamps of social media posts and actual hoax call events.
 By identifying patterns in the timing of posts, we were
 able to assess whether social media activity increased
 in the lead-up to hoax events, suggesting coordinated
 behavior.
 A time series analysis was performed to detect bursts of
 activity around specific times or events.
 E. Network Analysis
 • User Network Mapping: A user interaction network was
 built based on user mentions, replies, and shared content.
 Each user was represented as a node, and interactions
 (e.g., mentions, replies) were represented as edges con
necting the nodes.
 This network was analyzed to identify clusters of users
 who are frequently interacting with each other, which
 may indicate coordinated behavior typical of hoax cam
paigns.
 • Community Detection: The Louvain method was used
 for community detection within the user interaction net
work. This method is effective for identifying communi
ties within large networks, allowing us to pinpoint groups
 of users who may be coordinating hoax activities.
 The Louvain method uses modularity optimization to
 partition the network into communities where the density
 of edges within communities is maximized, and the
 density between communities is minimized.
 After detecting communities, we analyzed the behavior
 and content shared within each community to identify
 potential hoax-related groups.
 Fig. 1: System Workflow Diagram
 • Data Collection: Social media APIs (PRAW, Wikipedia
Scraper) are used to collect posts from Reddit and
 Wikipedia.
 • Preprocessing: Text is cleaned, tokenized, and
 lemmatized using NLTK and SpaCy.
 • Analysis:
 1. SGD Classifier: Content analysis to classify posts as
 hoax-related or not.
 2. VADER Sentiment Analysis: Sentiment analysis to
 assess the potential threat level.
 3. Temporal Analysis: Correlation between post times
tamps and hoax events.
 4. Network Analysis: User interaction network creation
 and community detection using Louvain method.
 • Visualization: Interactive visualizations of the user inter
action network using pyvis and Gephi.
 IX. KEY FEATURES IMPLEMENTED
 • Multi-platform Data Collector: Collects data from Red
dit and Wikipedia using respective APIs.
 • Content Analysis using SGD: Uses SGD classifier to
 detect hoax-related posts.
 • Sentiment Analysis with VADER: Analyzes the sen
timent of posts using the VADER sentiment analysis
 technique.
 • User Network Mapping: Builds a user interaction net
work to detect potential coordination in hoax campaigns.
 • Community Detection via Louvain: Uses the Louvain
 method for detecting communities of users potentially
 involved in hoax campaigns.
 • Interactive Dashboard using PyVis and Gephi: Visu
alizes the user interaction networks and detected commu
nities using pyvis and Gephi.
X. VISUALIZATIONS & RESULTS
 D. Temporal Trends
 A. Sentiment Dashboard
 Fig. 2: Sentiment Distribution Across Posts
 B. Author Network Visualization
 Fig. 3: User Interaction Network with Communities
 C. Content Analysis
 Fig. 4: Temporal Trends in Hoax Posts
 Fig. 5: Temporal Trends in Hoax Posts
 E. Author Network Evolution
 Fig. 6: Temporal Trends in Hoax Posts
 • Stage 1: The network is sparse and highly fragmented,
 with isolated community clusters and low interaction.
 • Stage 2: Slight increase in connectivity as a few nodes
 begin bridging smaller communities.
 • Stage 3: More cross-community links appear, indicating
 growing interaction and discussion spread.
 • Stage 4: A few central authors (e.g., KiwiDillwrites)
 become key connectors, suggesting the emergence of
 influential spreaders in the network.
 XI. SUGGESTIONS
 • Enhance the classifier by incorporating more advanced
 models such as BERT for better contextual understanding
 of social media content.
 • Improve the temporal analysis by considering external
 events and their correlation with hoax-related posts.
 • Optimize community detection by exploring additional
 algorithms such as the Girvan-Newman method.
XII. CHALLENGES FACED
 • Data Collection: Meta Platforms (Facebook, Instagram):
 Meta does not allow scraping of data from their platforms,
 which posed a significant challenge in gathering social
 media data. The restrictions made it difficult to collect a
 wide range of posts for analysis.
 Twitter API: Twitter’s Academic API access required a
 paid subscription, costing around $200 per month, which
 added a financial constraint to the project. This limitation
 hindered the ability to collect a larger volume of data over
 time.
 • Classification Model Performance: SGD classifier per
formed well, but the model could be further improved
 by incorporating more diverse features, such as post
 metadata (e.g., user reputation, number of followers).
 • Network Detection: Detecting meaningful communities
 in large social media networks was computationally
 expensive and required careful tuning of the Louvain
 method parameters.
 • Real-Time Analysis: Handling real-time data collection
 and processing posed scalability challenges, especially
 when dealing with high volumes of social media data.
 XIII. EVALUATION METRICS
 • HoaxDetection: Precision, recall, and F1-score for hoax
related post classification.
 The classifier achieved an accuracy of 75%.
 • Community Detection: Evaluated the effectiveness of
 the Louvain method in detecting meaningful communities
 by comparing the detected communities to known hoax
related user groups.
 • Sentiment Analysis: VADER sentiment analysis was
 evaluated based on the correlation of sentiment scores
 with hoax detection outcomes.
 • Network Analysis: Evaluated the user network and com
munity detection results using modularity and community
 coherence metrics.
 XIV. CONCLUSION AND FUTURE WORK
 • Future Work: Expand the system to include real-time
 data collection and analysis.
 Integrate more sophisticated machine learning models
 (e.g., BERT, GPT) for improved content classification.
 Explore deeper integration of multi-modal data (e.g.,
 images, videos) to enhance detection capabilities.
 Improve network detection by combining community
 detection with additional network analysis techniques.
 • Conclusion: The system provides a comprehensive solu
tion for detecting and analyzing hoax call campaigns on
 social media.
 By combining sentiment analysis, content classification,
 network analysis, and temporal correlation, the system
 can effectively detect coordinated hoax campaigns and
 assess their threat levels.
 Future improvements will focus on scalability, real-time
 processing, and more advanced machine learning tech
niques.
 REFERENCES
 [1] Understanding the Evolution of Reddit in Temporal ...- ResearchGate,
 accessed on April 24, 2025. Available at: https://www.researchgate.net
 /profile/Daniel-Schroeder-11/publication/366041622 Understanding
 t
 he Evolution of Reddit in Temporal Networks induced by User A
 ctivity/links/638f38fe11e9f00cda21b131/Understanding-the-Evolution-of-Reddit-in-Temporal-Networks-induced-by-User-Activity.pdf
 [2] (PDF) Understanding the Evolution of Reddit in Temporal Networks ...,
 accessed on April 24, 2025. Available at: https://www.researchgate.net
 /publication/366041622 Understanding the Evolution of Reddit in T
 emporal Networks induced by User Activity
 [3] Subreddit Links Drive Community Creation and User Engagement on
 Reddit- arXiv, accessed on April 24, 2025. Available at: https://arxiv.
 org/pdf/2203.10155
 [4] Retracing the evolution of Reddit through post data : r/TheoryOfReddit,
 accessed on April 24, 2025. Available at: https://www.reddit.com/r/The
 oryOfReddit/comments/1a7aoj/retracing the evolution of reddit throu
 gh post/
 [5] Fakeddit, accessed on April 24, 2025. Available at: https://fakeddit.net
 lify.app/
 [6] aclanthology.org, accessed on April 24, 2025. Available at: https://acla
 nthology.org/2020.lrec-1.755.pdf
 [7] [1911.03854] r/Fakeddit: A New Multimodal Benchmark Dataset for
 Fine-grained Fake News Detection- arXiv, accessed on April 24, 2025.
 Available at: https://arxiv.org/abs/1911.03854
 [8] Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake
 News Detection, accessed on April 24, 2025. Available at: https://acla
 nthology.org/2020.lrec-1.755/
 [9] Fake News Detection Using Stochastic Gradient Descent Algorithm
iosrjen, accessed on April 24, 2025. Available at: https://iosrjen.org/Pa
 pers/Conf.19018-2019/INFT/Volume-5/5.%2012-14.pdf
 [10] kapilsinghnegi/Fake-News-Detection: This project detects whether a
 news is fake or not using machine learning.- GitHub, accessed on April
 24, 2025. Available at: https://github.com/kapilsinghnegi/Fake-News-D
 etection
 [11] Fake News Detection using Machine Learning- GeeksforGeeks, ac
cessed on April 24, 2025. Available at: https://www.geeksforgeeks.org/
 fake-news-detection-using-machine-learning/
 [12] VADER sentiment analysis (with examples)- Hex, accessed on April
 24, 2025. Available at: https://hex.tech/templates/sentiment-analysis/va
 der-sentiment-analysis/
 [13] Sentiment Analysis Using VADER- Analytics Vidhya, accessed on
 April 24, 2025. Available at: https://www.analyticsvidhya.com/blog
 /2022/10/sentiment-analysis-using-vader/
 [14] What is Vader Sentiment Analysis- VizRefra, accessed on April 24,
 2025. Available at: https://www.vizrefra.com/sentiment-analysis/what-v
 ader-sentiment-analysis/
 [15] (PDF) Understanding Sentiment Analysis with VADER: A ..., accessed
 on April 24, 2025. Available at: https://www.researchgate.net/publicati
 on/381650914 Understanding Sentiment Analysis with VADER
 A
 Comprehensive Overview and Application
 [16] Real-Time Sentiment Insights from X Using VADER, DistilBERT, and
 Web-Scraped Data, accessed on April 24, 2025. Available at: https:
 //arxiv.org/html/2504.15448v1
 [17] Understanding Sentiment Analysis: What It Is and Why It Matters
Webuters Technologies, accessed on April 24, 2025. Available at: https:
 //www.webuters.com/understanding-sentiment-analysis-what-it-is-and-why-it-matters
 [18] AICyberInnovate Spectrum Magazine Understanding Sentiment Analy
sis with VADER: A Comprehensive Guide, accessed on April 24, 2025.
 Available at: https://aicybersecuritycenter.com/wp-content/uploads/202
 3/11/Article-9-VADER.pdf
 [19] (PDF) Co-Comment Network: A Novel Approach to Construct Social
 ..., accessed on April 24, 2025. Available at: https://www.researchgate
 .net/publication/360081517 Co-Comment Network
 A
 Novel Approac
 h to Construct Social Networks within Reddit
 [20] Co-Comment Network: A Novel Approach for Construction of Social
 ..., accessed on April 24, 2025. Available at: https://www.scielo.org.m
 x/scielo.php?script=sci arttext&pid=S1405-55462022000100311
[21] Co-Comment Network: A Novel Approach to Construct Social Networks
 within Reddit — Kanti Baowaly — Computaci´on y Sistemas, accessed
 on April 24, 2025. Available at: https://www.cys.cic.ipn.mx/ojs/index.p
 hp/CyS/article/view/4175
 [22] ANovel Approach to Construct Social Networks within Reddit, accessed
 on April 24, 2025. Available at: https://www.scielo.org.mx/pdf/cys/v2
 6n1/2007-9737-cys-26-01-311.pdf
 [23] Co-Comment Network: A Novel Approach for Construction of Social
 Networks within Reddit, accessed on April 24, 2025. Available at: https:
 //www.scielo.org.mx/scielo.php?script=sci abstract&pid=S1405-55462
 022000100311&lng=es&nrm=iso
 [24] A social network caught in the Web — Request PDF- ResearchGate,
 accessed on April 24, 2025. Available at: https://www.researchgate.net
 /publication/220167109 A social network caught in the Web
 [25] www.researchgate.net, accessed on April 24, 2025. Available at: https:
 //www.researchgate.net/profile/Nguyen-Dong-28/publication/34819476
 8 Graph Neural Network Combined Knowledge Graph for Recom
 mendation System/links/63a2afa69835ef259037e010/Graph-Neural-N
 etwork-Combined-Knowledge-Graph-for-Recommendation-System.pdf
 [26] Human-Centric Community Detection in Hybrid Metaverse Networks
 with Integrated AI Entities — OpenReview, accessed on April 24, 2025.
 Available at: https://openreview.net/forum?id=aq393AkrKa&referrer=
 %5Bthe%20profile%20of%20Ya-Wen%20Teng%5D(%2Fprofile%3Fid
 %3D∼Ya-Wen Teng1)
 [27] Human-Centric Community Detection in Hybrid Metaverse Networks
 with Integrated AI Entities- arXiv, accessed on April 24, 2025. Available
 at: https://arxiv.org/html/2502.10750v1
 [28] Analysis of Algorithms for Detecting Users’ Behavioral Models based
 on Sessions Data, accessed on April 24, 2025. Available at: https://cs
 imq-journals.rtu.lv/csimq/article/download/csimq.2024-41.04/268/468
 [29] (PDF) Hybrid Louvain-Clustering Model Using Knowledge Graph for
 ..., accessed on April 24, 2025. Available at: https://www.researchgate
 .net/publication/351518950 Hybrid Louvain-Clustering Model Using
 Knowledge Graph for Improvement of Clustering User’s Behavio
 r on Social Networks
 [30] Louvain Clustering : r/bioinformatics- Reddit, accessed on April 24,
 2025. Available at: https://www.reddit.com/r/bioinformatics/comments
 /hwhjo0/louvain clustering/
 [31] Louvain algorithm for graph clustering gives completely different result
 when running in Spark/Scala and Python, why is that happening?- Stack
 Overflow, accessed on April 24, 2025. Available at: https://stackoverflo
 w.com/questions/55601566/louvain-algorithm-for-graph-clustering-giv
 es-completely-different-result-when-ru
 [32] A Comprehensive Review of Community Detection in Graphs- arXiv,
 accessed on April 24, 2025. Available at: https://arxiv.org/html/2309.11
 798v5
 [33] A/ Prof. Hai V. Pham (0000-0001-8325-1662)- ORCID, accessed on
 April 24, 2025. Available at: https://orcid.org/0000-0001-8325-1662
 [34] Quantifying community evolution in temporal networks- arXiv, ac
cessed on April 24, 2025. Available at: https://arxiv.org/html/2411.
 10632v2
 [35] CS224W Project Milestone Report Upon the Advent of Eternal Septem
ber: a Case Study on Reddit Communities’ Latent Networks, accessed
 on April 24, 2025. Available at: http://snap.stanford.edu/cs224w-17-dat
 a/best-milestones-16/cs224w-milestone.pdf
 [36] Assessing temporal and spatial features in detecting disruptive users
 on Reddit--ORCA- Cardiff University, accessed on April 24, 2025.
 Available at: https://orca.cardiff.ac.uk/id/eprint/136908/7/Detecting dis
 ruptive behaviour on Reddit ASONAM .pdf
 [37] Temporal dynamics of coordinated online behavior: Stability, archetypes,
 and influence — PNAS, accessed on April 24, 2025. Available at: https:
 //www.pnas.org/doi/10.1073/pnas.2307038121
 [38] Social Network Datasets Archives- Complex Adaptive Systems Labo
ratory, accessed on April 24, 2025. Available at: https://complexity.cec
 s.ucf.edu/social-network-datasets/
 [39] 2025 Research Projects- The Science Internship Program (SIP), ac
cessed on April 24, 2025. Available at: https://sip.ucsc.edu/2025-resea
 rch-projects/
 [40] [1911.03854] r/Fakeddit: A New Multimodal Benchmark Dataset for
 Fine-grained Fake News Detection- ar5iv, accessed on April 24, 2025.
 Available at: https://ar5iv.labs.arxiv.org/html/1911.03854
 [41] r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained
 Fake News Detection, accessed on April 24, 2025. Available at: https:
 //www.researchgate.net/publication/337184096 rFakeddit A New Mul
 timodal Benchmark Dataset for Fine-grained Fake News Detection
 [42] Fakeddit Dataset- Papers With Code, accessed on April 24, 2025.
 Available at: https://paperswithcode.com/dataset/fakeddit
 [43] r/Fakeddit New Multimodal Benchmark Dataset for Fine-grained Fake
 News Detection- GitHub, accessed on April 24, 2025. Available at:
 https://github.com/entitize/Fakeddit
 [44] r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained,
 accessed on April 24, 2025. Available at: https://scite.ai/reports/r-faked
 dit-a-new-multimodal-benchmark-jMn8LDRj
 [45] Publication: r/Fakeddit: A New Multimodal Benchmark Dataset for,
 accessed on April 24, 2025. Available at: https://cs.reviewer.ly/app
 /publication/f0b349ea-775a-445f-8bcd-afc632266df2
 [46] Novel approach for predicting fake news stance detection using large
 word embedding blending and customized CNN model — PLOS One,
 accessed on April 24, 2025. Available at: https://journals.plos.org/plos
 one/article?id=10.1371/journal.pone.0314174
 [47] Fake News Detection Using LSTM Neural Network Augmented with
 SGD Classifier, accessed on April 24, 2025. Available at: https://solids
 tatetechnology.us/index.php/JSST/article/view/4504
 [48] Specathon2k21/Team-Brighter-2021-Fake-Reviews-Detection: Fake re
views detection using SGD Classifier , with an flexible user interface
GitHub, accessed on April 24, 2025. Available at: https://github.com/S
 pecathon2k21/Team-Brighter-2021-Fake-Reviews-Detection
 [49] Are Strong Baselines Enough? False News Detection with Machine
 Learning- MDPI, accessed on April 24, 2025. Available at: https:
 //www.mdpi.com/1999-5903/16/9/322
 [50] Comparison of Machine Learning and Deep Learning Algorithms in De
tecting Fake News- International Institute of Informatics and Systemics,
 accessed on April 24, 2025. Available at: https://www.iiis.org/CDs2024
 /CD2024Summer//papers/SA029NJ.pdf
 [51] arxiv.org, accessed on April 24, 2025. Available at: https://arxiv.org/pd
 f/2203.09936
 [52] A comparison of artificial intelligence models used for fake news
 detection- The Distant Reader, accessed on April 24, 2025. Available
 at: https://distantreader.org/stacks/journals/bulletin/bulletin-1680.pdf
 [53] dedeepyay/Fake-Reviews-Detection- GitHub, accessed on April 24,
 2025. Available at: https://github.com/dedeepyay/Fake-Reviews-D
 etection
 [54] Detecting Fake News in Social Media Using Voting Classifier- Re
searchGate, accessed on April 24, 2025. Available at: https://www.rese
 archgate.net/publication/356711511 Detecting Fake News in Social
 Media Using Voting Classifier
 [55] (PDF) Hoax News Classification using Machine Learning Algorithms,
 accessed on April 24, 2025. Available at: https://www.researchgate.net
 /publication/364059463 Hoax News Classification using Machine L
 earning Algorithms/download
 [56] From Misinformation to Insight: Machine Learning Strategies for Fake
 News Detection, accessed on April 24, 2025. Available at: https://ww
 w.mdpi.com/2078-2489/16/3/189
 [57] (PDF) Hoax News Classification Using Machine Learning Algorithms,
 accessed on April 24, 2025. Available at: https://www.researchgate.net
 /publication/338036394 Hoax News Classification Using Machine L
 earning Algorithms
 [58] A novel approach to fake news classification using LSTM-based deep
 learning models, accessed on April 24, 2025. Available at: https://pmc.
 ncbi.nlm.nih.gov/articles/PMC10800750/
 [59] Fake News Classification Using Machine Learning– IJERT, accessed
 on April 24, 2025. Available at: https://www.ijert.org/fake-news-classif
 ication-using-machine-learning
 [60] Comparative Analysis of Machine Learning Algorithms for Detecting
 Fake News: Efficacy and Accuracy in the Modern Information Ecosys
tem, accessed on April 24, 2025. Available at: https://www.jurnal.itsci
 ence.org/index.php/CNAPC/article/view/3466
 [61] Full article: Optimized Fake News Classification: Leveraging Ensembles
 Learning and Parameter Tuning in Machine and Deep Learning Methods,
 accessed on April 24, 2025. Available at: https://www.tandfonline.com/
 doi/full/10.1080/08839514.2024.2385856
 [62] How can a ML algorithm learn to classify fake news?- Data Science
 Stack Exchange, accessed on April 24, 2025. Available at: https://data
 science.stackexchange.com/questions/81797/how-can-a-ml-algorithm-l
 earn-to-classify-fake-news
 [63] (PDF) Some Like it Hoax: Automated Fake News Detection in Social
 ..., accessed on April 24, 2025. Available at: https://www.researchgate
.net/publication/316471370 Some Like it Hoax Automated Fake N
 ews Detection in Social Networks
 [64] Some Like it Hoax Dataset- Papers With Code, accessed on April 24,
 2025. Available at: https://paperswithcode.com/dataset/some-like-it-hoa
 x
 [65] Some Like it Hoax: Automated Fake News Detection in Social Networks- Luca de Alfaro, accessed on April 24, 2025. Available at: https://luca
 .dealfaro.com/papers/17/UCSC-SOE-17-05.pdf
 [66] Some Like it Hoax: Automated Fake News Detection in Social Networks- Luca de Alfaro, accessed on April 24, 2025. Available at: https://luca
 .dealfaro.com/papers/17/some like it hoax.pdf
 [67] Fake News Detection — Papers With Code, accessed on April 24, 2025.
 Available at: https://paperswithcode.com/task/fake-news-detection?pag
 e=3&q=
 [68] Some Like it Hoax: Automated Fake News Detection in Social ..., ac
cessed on April 24, 2025. Available at: https://arxiv.org/pdf/1704.07506
 [69] Code for the paper ”Some Like it Hoax: Automated Fake News
 Detection in Social Networks” by E.Tacchini, G.Ballarin, M.L.Della
 Vedova, S.Moret and L.De Alfaro (2017)- GitHub, accessed on April
 24, 2025. Available at: https://github.com/gabll/some-like-it-hoax
 [70] [1704.07506] Some Like it Hoax: Automated Fake News Detection in
 Social Networks- arXiv, accessed on April 24, 2025. Available at: https:
 //arxiv.org/abs/1704.07506
 [71] ELG- Some Like it Hoax- European Language Grid, accessed on April
 24, 2025. Available at: https://live.european-language-grid.eu/catalogue
 /corpus/5091
 [72] Automated Fake News Detection in the Age of Digital Libraries,
 accessed on April 24, 2025. Available at: https://ital.corejournals.or
 g/index.php/ital/article/view/12483
 [73] Youvan VADER Sentiment Analysis — Download Scientific Diagram
ResearchGate, accessed on April 24, 2025. Available at: https://www.re
 searchgate.net/figure/Youvan-VADER-Sentiment-Analysis fig1 38165
 0914
 [74] cjhutto/vaderSentiment: VADER Sentiment Analysis ...- GitHub, ac
cessed on April 24, 2025. Available at: https://github.com/cjhutto/vader
 Sentiment
 [75] View of VADER: A Parsimonious Rule-Based Model for Sentiment
 Analysis of Social Media Text, accessed on April 24, 2025. Available
 at: https://ojs.aaai.org/index.php/ICWSM/article/view/14550/14399 """

    results = evaluate(pdf_path, reference_text)
    for lib, metrics in results.items():
        print(f"\n{lib}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
