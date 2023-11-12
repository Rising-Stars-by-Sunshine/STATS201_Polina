# 1.1. Understanding Literature at the Frontiers of Machine Learning for Social Science

## Identifying propaganda from online social networks during COVID-19 using machine learning techniques

![Whimsical Chart](https://static.us.edusercontent.com/files/ZacuVSVeAYum0N2wzAvRSvZI)
Created with the help of Whimsical

### Background/Motivation:

The social media networks became the only means of communication during the COVID-19. The extent of information shared during that period has created a whole discussion about the validity of statements shared online, the purpose behind them, and what effect they have to have on the audience. The propaganda not medically proven regarding the virus spread, cure, and vaccination has been extensive, and the article looks into whether machine learning can be used to mark unverified claims and propaganda regarding COVID-19 online. (Khanday 2021, 1-2)

### Research Question:

How can machine learning techniques be used to identify propaganda online?

### Application Scenarios:

The results can be applied to a multitude of contexts and situations of interaction on social media networks to determine what features propagandist texts share and how machine learning algorithms can be trained to determine the text's purpose (propaganda or not).

### Methodology:

The methodology bases on a four-step system illustrated in Figure 1: data collection, preprocessing, feature engineering, and classification. The researcher first performed Twitter data collection through its application program interface (API) through the trending hashtags related to the research topic and provided manual annotations. (Khanday 2021, 3) During the data preprocessing stage, the textual data was tokenized, stop words were introduced, and the tokenized words went through lemmatization (discovering an intended meaning and connotation of words). Feature engineering utilized the TF/IDF technique, bag of words, and tweet length techniques and determined commonly used combinations containing one, two, or three words within the twits. The classification of data into propaganda and non-propaganda classes was performed through logistic regression, multinomial naive Bayes, support vector machine, and decision tree. (Khanday 2021, 3-4)

![Figure 1](https://static.us.edusercontent.com/files/LuZVh29feuXq4ZbknzGDZCvY)
Figure 1. A proposed system for identifying propaganda on online social networks. (Khanday 2021, 3)

### Results:

The research showed that Twitter posts marked as 'propaganda' by the algorithm contained more words and were lengthier. The research recognizes that the data used for model training had a primary connection with the COVID-19 pandemic, and the application and use of more information from a broad range of sectors will provide a more extensive analysis of online propaganda. Additionally, more feature engineering techniques will help achieve better text categorization.

### Intellectual Merits/Practical Impacts:

The study successfully proved that machine learning algorithms have the potential to provide a social media network analysis to determine the purpose of the statements through shared characteristics and sets of words. The study may be further enhanced by resorting to the use of more features as well as incorporating Deep learning algorithms.

### Bibliography

Khanday, Akib Mohi, Qamar Rayees Khan, and Syed Tanzeel Rabani. Identifying Propaganda from Online Social Networks during COVID-19 Using Machine Learning Techniques. International Journal of Information Technology, no. 13 (2021): 115–122. https://doi.org/10.1007/s41870-020-00550-5. 

### Writing Accuracy
<img width="184" alt="Screenshot 2023-11-12 at 7 04 34 PM" src="https://github.com/Rising-Stars-by-Sunshine/STATS201_Polina/assets/148934457/b4b61a24-aed5-47f3-a765-f69fabcdef58">
<img width="500" alt="Screenshot 2023-11-12 at 7 04 46 PM" src="https://github.com/Rising-Stars-by-Sunshine/STATS201_Polina/assets/148934457/5bb1f91c-cb7f-4c7a-83af-1bc125a47dbe">
<img width="318" alt="Screenshot 2023-11-12 at 7 09 24 PM" src="https://github.com/Rising-Stars-by-Sunshine/STATS201_Polina/assets/148934457/444e985d-762e-4504-8d98-33ac3db2dcb3">
*The similarity report highlighted the use of sources, title of the paper, and techniques used for algorithm training. Everything was cited in accordance with Chicago Manual

# 1.2. Identifying Research Ideas at the Frontiers of Machine Learning for Social Science

## Applying Machine Learning Algorithms for Detecting Misinformation Online

![Chart](https://static.us.edusercontent.com/files/fw4G1WxxtdleJldr6Pd4qsLo)
Created with the help of Markmap

### Background/Motivation

In the modern world, the internet has not only become an important tool for interpersonal communication but also a primary channel for information dissemination. This trend has only accelerated over the recent years, particularly in the wake of the COVID-19 pandemic, as more people turned to social media platforms for information presented in a wide range of content formats, including text, images, videos, and audio. With an extensive integration of information into the online sphere, the critical issue of verifying the authenticity of the content has emerged as a central concern for individual users seeking reliable information and policymakers determined to preserve the integrity of public discussions online and decision-making processes. It underscores the urgency of developing effective strategies and technologies to combat misinformation dissemination and ensure the reliability of the digital information ecosystem. One of the possible methods to address the misinformation spread is the application of machine learning algorithms for misinformation detection.

### Research Question

How can machine learning algorithms be applied to misinformation detection online?

### Application Scenarios

The potential applications of machine learning algorithms are vast, but some central ones include the following spheres. A trained model for misinformation detection can assist news outlets in fact-checking and identifying potentially false information before it is published. Additionally, misinformation can have a significant impact on political discourse and elections. Therefore, machine learning can be used to monitor and address false claims and deceptive political advertising. Finally, misinformation can also pertain to cultural and social issues, and machine learning algorithms can help identify and address false information related to sensitive topics.

### Methodology

The research will take into account the existing findings and trained models for data detection proposed by Fatemeh Torabi Asr and Maite Taboada (2019) and Nayeon Lee, Belinda Z. Li, Sinong Wang, Pascale Fung, Hao Ma, Wen-tau Yih, Madian Khabsa (2021). The machine learning algorithm developed as a result of this research will be able to detect the presence of news bias, clickbait posts, coverage of fake news, and spreading rumors, as well as detect whether the post is intended to be satirical or not, and if the information present is up-to-date. The first four categories have been trained within the Unified Misinfo Model (UNIFIEDM2) (Lee et al. 2021), while the latter two categories have not been applied for algorithm training. The model proposed would be based on BERT (Bidirectional Encoder Representations from Transformers) with Multi-Label Classification, as well as Multi-Task Learning (MTL). The algorithm training will be based on binary classification of each information category (biased/objective, clickbait/not-clickbait, fake/true, rumor/fact, satire/not-satire, outdated/up-to-date).

### Results

The developed misinformation detection model expects to achieve the following results: 

- Determining the algorithms better suited for the machine learning algorithm; 
- Analyzing the preliminary data interpretation and detecting refinery areas; 
- Providing an accurate detection of the classification of the information; 
- Evaluation of the common trends shared between the data.

### Intellectual Merits/Practical Impacts

The research contributes to a plethora of fields, including, but not limited to policymaking and social media, by refining the information received and presented as factual by applying machine learning algorithms to predict the validity of information. Some of the practical impacts include:

- Improved Information Quality;
- Trust in Online Information;
- Enhanced Media Literacy.

### Bibliography
Lee, Nayeon, Belinda Z. Li, Sinong Wang, Pascale Fung, Hao Ma, Wen-tau Yih, and Madian Khabsa. "On unifying misinformation detection." arXiv preprint arXiv:2104.05243 (2021).

Torabi Asr, Fatemeh, and Maite Taboada. "Big Data and quality data for fake news and misinformation detection." Big Data & Society 6, no. 1 (2019): 2053951719843310.

### Writing Accuracy
<img width="188" alt="Screenshot 2023-11-12 at 6 58 46 PM" src="https://github.com/Rising-Stars-by-Sunshine/STATS201_Polina/assets/148934457/c164b083-f530-437c-b610-e2fbb9dd442a">
<img width="493" alt="Screenshot 2023-11-12 at 6 58 56 PM" src="https://github.com/Rising-Stars-by-Sunshine/STATS201_Polina/assets/148934457/79be564b-7ed7-411d-981a-988128046658">
<img width="320" alt="Screenshot 2023-11-12 at 7 15 08 PM" src="https://github.com/Rising-Stars-by-Sunshine/STATS201_Polina/assets/148934457/2a2949c1-0339-4d09-9523-152dc2ce5380">
