import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, gutenberg, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# Example of tokenizing
example_text = 'Hi there, how are you doing today. Python is awesome. The sky is blue.'
print(sent_tokenize(example_text))


#Example of how to filter out stop words
example_sent = 'This is an example showing off stop word filtration.'
stop_words = set(stopwords.words('english'))

words = word_tokenize(example_sent)

# filtered_sentence  = []
# for w in words:
# 	if w not in stop_words:
# 		filtered_sentence.append(w)
filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)


#Stemming
ps = PorterStemmer()
# example_words = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly']
new_text = 'It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once.'
# for w in example_words:
# 	print(ps.stem(w))
words = word_tokenize(new_text)
for w in words:
	print(ps.stem(w))


#Part of Speech Tagging, Chunking, Chinking
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)
# def process_content():
# 	try:
# 		for i in tokenized:
# 			words = nltk.word_tokenize(i)
# 			tagged = nltk.pos_tag(words)
# 			chunkGram = '''Chunk: {<.*>+}
# 			}<VB.?|IN|DT>+{''' #regular expressions
# 			chunkParser = nltk.RegexpParser(chunkGram)
# 			chunked = chunkParser.parse(tagged)
# 			print(chunked)
# 	except Exception as e:
# 		print(str(e))
# process_content()


#Name Entity Recognition
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)
# def process_content():
# 	try:
# 		for i in tokenized:
# 			words = nltk.word_tokenize(i)
# 			tagged = nltk.pos_tag(words)
# 			named_ent = nltk.ne_chunk(tagged)
# 	except Exception as e:
# 		print(str(e))
# process_content()



#Lemmatizing: liek stemming but return a real word, the default is pos='n' for noun
lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize('better'))
# print(lemmatizer.lemmatize('better', pos='a'))



#Corpora
sample = gutenberg.raw('bible-kjv.txt')
tok = sent_tokenize(sample)
# print(tok[5:15])



# WordNet
syns = wordnet.synsets('program') #gets syns
print(syns)
syns[0].definition() #prints the definition
syns[0].examples() #gives an example of the word used in sentence



