import rouge
import os
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

def evaluate_lead3(file_name):
	# Read the story file.
	story_file = open(file_name, 'r', encoding='ISO-8859-1')
	# Lowercase everything.
	lines = [line.strip().lower() for line in story_file.readlines()]

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

	sents_file = open(file_name[:-6]+'.sents.txt', 'r', encoding='ISO-8859-1')
	article_lines = [line.strip().lower() for line in sents_file.readlines()]
	limit = 275
	evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
							max_n=2,
							limit_length=True,
							length_limit=limit,
							length_limit_type='bytes',
							apply_avg=True,
							apply_best=None,
							alpha=0.5,	# Default F1_score
							weight_factor=1.2,
							stemming=True)

	# Joining all article sentences together into a string.
	lead3_summary = normalize(' '.join(article_lines[:3]))
	gold_summary = normalize(' '.join(highlights))
	scores = evaluator.get_scores(lead3_summary, [gold_summary])

	return scores['rouge-1']['r'], scores['rouge-2']['r'], scores['rouge-l']['r']

def evaluate_file(file_name, method):
	print( '--------------------            ', method, '                 -------------------')
	# Read the story file.
	story_file = open(file_name, 'r', encoding='ISO-8859-1')
	label_file = open(file_name[:-len('.story')] + method, 'r', encoding='ISO-8859-1')
	# Lowercase everything.
	lines = [line.strip().lower() for line in story_file.readlines()]

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

	sents_file = open(file_name[:-6]+'.sents.txt', 'r', encoding='ISO-8859-1')
	article_lines = [line.strip().lower() for line in sents_file.readlines()]
	lines = [(float(label.strip()),i) for i,label in enumerate(label_file.readlines()[:len(article_lines)]) ]
	lines.sort(key=lambda x: (-x[0], x[1]))
	limit = 275
	evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
							max_n=2,
							limit_length=True,
							length_limit=limit,
							length_limit_type='bytes',
							apply_avg=True,
							apply_best=None,
							alpha=0.5,	# Default F1_score
							weight_factor=1.2,
							stemming=True)

	# Joining all article sentences together into a string.
	gold_summary = normalize(' '.join(highlights))
	predicted_summary = ' '.join([article_lines[line[1]]for line in lines])[:limit]
	#print('1---',predicted_summary)
	#print('2---',gold_summary)
	scores = evaluator.get_scores(predicted_summary, [gold_summary])


	return scores['rouge-1']['r'], scores['rouge-2']['r'], scores['rouge-l']['r']
i = 0
tr1 = tr2 = trl = 0
for story_file in glob.glob('data/test/*.story'):
	i+=1
	r1, r2, rl = evaluate_file(story_file, '_SummaRunnerScores.txt')
	# r1, r2, rl = evaluate_lead3(story_file)
	#print(i, r1, r2, rl)
	tr1 += r1
	tr2 += r2
	trl += rl
		
print('tr1: ', tr1/i)
print('tr2: ', tr2/i)
print('trl: ', trl/i)
