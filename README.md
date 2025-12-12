# I Think You Should Leave
### Sentiment analysis on the faces and transcripts from the Netflix show

Question: Can machine learning models understand cringe comedy?

The answer: I don't think so.

:warning: Content warning: Explicit language in some of the files.

---

I sat down and watched all 3 seasons (18 episodes, 86 sketches) of one of my favourite TV shows: *I Think You Should Leave with Tim Robinson* with a bunch of machine learning models. I was curious how machines would interpret a cringe comedy sketch show. 

## Methodology

1. Scrape metadata (sketch-level data, including time stamps)
2. Prepare data (transcripts and videos)
3. Text classification for sentiments
4. Face detection
5. Emotion classification

## 1. Scraping the metadata

[`metadata_scrape.ipynb`](https://github.com/jfung53/itysl/blob/main/metadata_scrape.ipynb)  
I found a Netflix blog post with all the timestamps and names of each sketch. I used this to create a [`metadata file`](https://github.com/jfung53/itysl/blob/main/episode_metadata.csv) that would be used to process the video data. 

While re-watching the show, I assigned my own category to each of the sketches and added it to the metadata file. The top three are office scenarios, parties, and TV commercials. 

## 2. Preparing the data

For the [`transcripts`](https://github.com/jfung53/itysl/blob/main/transcripts.json) I wasn't able to scrape the site, but it was easy enough to copy and paste from the [I Think You Should Leave database](https://www.itysldb.com/). I cleaned up the transcripts using the NLP library `spaCy` to identify and extract full sentences, see [`transcripts_cleaning.ipynb`](https://github.com/jfung53/itysl/blob/main/transcripts_cleaning.ipynb). 

## 3. Text Sentiments

Colab notebook: [`sentiment_transcripts.ipynb`](https://colab.research.google.com/drive/1GuUbnw1pVMQrNDVKJ98bJGYvR3YWWB_D?usp=sharing)  
I tried a few models for sentiment/emotion classification. The first one was `DistilBERT` which only provided positive/negative results. I ended up choosing GoEmotions (`"SamLowe/roberta-base-go_emotions"`) which includes 28 emotions. The model found *25* emotions in the transcripts overall! The results are here: [`results_text_sentiments.csv`](https://github.com/jfung53/itysl/blob/main/results_text_sentiments.csv). Interestingly, the emotion it detected the most was 'neutral'. 

## 4. Face detection

Colab notebook: [`itysl_episode_capturing.ipynb`](https://colab.research.google.com/drive/1Q8I58HCuvhOhaOOsJz9wf2udh6g-EeOY)  
Next, I wanted to compare with video sentiments using a model called `InsightFace` to detect and track faces. First, I parsed the time stamps from `episode_metadata.csv` into seconds and then I ran some tests to determine the frame sampling rate, cosine similarity threshold, and minimum detection confidence. I decided somewhat arbitrarily to capture a face every 18 frames (just under 1 second). Capturing every frame would've taken way too long and I would have burnt through all of my student Colab credits, and 1 frame per second didn't feel like enough. Some emotional reactions felt faster than that. 

Initially, the cosine similarity threshold was set to `0.5` and it was identifying too many characters. For example, 16 characters were identified in a sketch that only had 4 people in it. Decreasing the threshold to `0.3` was perfect for getting the results I wanted. I watched a few sketches to verify that it correctly identified which faces belonged to the same person. I added some logic so I could interrupt the process and restart it if needed. Overall it took hours and lots of experimentation to get this the results I wanted! 

Since a lot of the background characters weren't often in clear view and didn't tend to add much to the plot I set the minimum detection confidence to `0.8` which worked well for ignoring low-confidence detections.  

There were still a lot of characters that were captured, so I manually reviewed the thumbnails which was pretty time consuming. Automating this probably would've taken an equally arduous amoutn of time, so I watched every episode... again. I tracked these in the [`character_review.csv`](https://github.com/jfung53/itysl/blob/main/character_review.csv) file and then converted it into a [`json format`](https://github.com/jfung53/itysl/blob/main/character_filters.json) to make it machine-parsable. I used this json to run some merge/keep/delete functions based on my manual review.  

## 5. Emotion classification

I searched long and hard for a facial expression model that would match the wide range of sentiments that `GoEmotions` could do. [This one](https://huggingface.co/ElenaRyumina/face_emotion_recognition) was the closest, I believe it does 27 emotions, but unfortunately I couldn't get it to work for me. 

I settled on [this vision transformer model](https://huggingface.co/mo-thecreator/vit-Facial-Expression-Recognition) that is trained on FER2013, MMI facial expressions, AffectNet datasets. It only classifies 7 facial emotions. Although the results were not as nuanced as the transcript results, the top emotion it classified was still 'neutral'. Extremely messy Colab notebook here: [`emotion_analysis.ipynb`](https://colab.research.google.com/drive/1SfXCZgZ4Q2P3KikqujSrnDJuz9eT9Kjq?usp=sharing#scrollTo=new_cell_6). I went all out with Google Gemini for this one, I asked it so many questions that it eventually told me that it was finished all the tasks instead of offering more help. 

## Closing Thoughts

Considering the topics that the sketches cover, most of them are a little mundane at face value: office interactions, parties full of middle-aged people, infomercials and traditional tv commercials, dating faux-pas, restaurant etiquette, driving... mostly everyday situations that we find ourselves in, made awkward. It's no wonder that most of the sentiments were 'neutral'. It would take some more advanced implementation to consider the context surrounding each sentence or individual face, and even then, the humour in a show like this can be extremely subjective, probably even more than other comedic genres. Most machine learning methods for identifying humour are based on pattern recognition and some even rely on the presence of a laugh track. The methods I used are nowhere near being able to understand a cringe comedy and what makes it funny. I find relief in that, because I think the writing is brilliant and I'd never want a machine to be able do this, anyway. 

---

**Data Sources**

Episode metadata: [Netflix article](https://www.netflix.com/tudum/articles/i-think-you-should-leave-sketch-names)\
Transcripts:  [I Think You Should Leave Database](https://www.itysldb.com/)

---

Pratt Institute Fall 2025 - INFO 656 Machine Learning Final Project

