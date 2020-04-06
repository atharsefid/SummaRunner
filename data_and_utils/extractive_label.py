import rouge
from nltk import tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import glob

stopset = set(stopwords.words('english')).union(string.punctuation)
wordnet_lemmatizer = WordNetLemmatizer()


def normalize(text):
	tokens = word_tokenize(text)
	tokens = [wordnet_lemmatizer.lemmatize(token.lower(), pos='n') for token in tokens if
			  token.lower() not in stopset]

	return ' '.join(tokens)


def label_sequence(story_file):
	# Read the story file.
	f = open(story_file, 'r', encoding='ISO-8859-1')

	# Lowercase everything.
	lines = [line.strip().lower() for line in f.readlines()]


	# Separate article and abstract sentences.
	article_lines = []
	highlights = []
	next_is_highlight = False

	for idx, line in enumerate(lines):

		if line == "":	# Empty line.
			continue

		elif line.startswith("@highlight"):
			next_is_highlight = True

		elif next_is_highlight:
			highlights.append(line)

		else:
			article_lines.append(line)

	# Joining all article sentences together into a string.
	article = ' '.join(article_lines)
	summary = ' '.join(highlights)

	sents = tokenize.sent_tokenize(article)
	labels = [0] * len(sents)

	evaluator = rouge.Rouge(metrics=['rouge-n'],
							max_n=2,
							limit_length=False,
							apply_avg=True,
							apply_best=None,
							alpha=0.5,	# Default F1_score
							weight_factor=1.2,
							stemming=True)
	cur_summary = ''
	prev_score = 0
	summary = normalize(summary)
	for i, sent in enumerate(sents):
		sent = normalize(sent)
		cur_summary += sent

		scores = evaluator.get_scores(cur_summary, [summary])
		# fscore = (scores['rouge-1']['f']+ scores['rouge-2']['f'])/2
		fscore = scores['rouge-1']['r']
		if fscore > prev_score:
			prev_score = fscore
			labels[i] = 1 # add sentence i to the summary
		else:
			cur_summary = cur_summary[:-len(sent)]
	with open(story_file[:-6]+ '.sents.txt', 'w') as sentFile:
		for sent in sents:
			sentFile.write(sent+'\n')
	with open(story_file[:-6]+ '.rouge1.txt', 'w') as labelFile:
		 for label in labels:
			 labelFile.write(str(label)+'\n')

i= 0 
for story_file in glob.glob('data/train/*.story'):
	i+=1
	print(i)
	label_sequence(story_file)
for story_file in glob.glob('data/test/*.story'):
	i+=1
	print(i)
	label_sequence(story_file)
for story_file in glob.glob('data/val/*.story'):
	i+=1
	print(i)
	label_sequence(story_file)
