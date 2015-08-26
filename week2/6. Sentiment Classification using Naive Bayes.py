import re
import os
import math
import elice_utils

def main():
    # Implement main function for Emotion Classifier
    training1_sentences = read_text_data('./txt_sentoken/pos/')
    training2_sentences = read_text_data('./txt_sentoken/neg/')
    testing_sentence = input()

    alpha = 0.1
    prob1 = 0.5
    prob2 = 0.5

    prob_pair = naive_bayes(training1_sentences, training2_sentences, testing_sentence, alpha, prob1, prob2)

    plot_title = testing_sentence
    if len(plot_title) > 50: plot_title = plot_title[:50] + "..."
    print(elice_utils.visualize_boxplot(plot_title,
                                        list(prob_pair),
                                        ['Positive', 'Negative']))

def naive_bayes(training1_sentence, training2_sentence, testing_sentence, alpha, prob1, prob2):
    # Exercise
    training_model1 = create_BOW(training1_sentence)
    training_model2 = create_BOW(training2_sentence)
    testing_model = create_BOW(testing_sentence)
    
    likelihood1 = calculate_doc_prob(training_model1, testing_model, alpha)
    likelihood2 = calculate_doc_prob(training_model2, testing_model, alpha)
    
    classify1 = math.log(prob1) + likelihood1
    classify2 = math.log(prob2) + likelihood2
    
    return normalize_log_prob(classify1, classify2)

def read_text_data(directory):
    # We already implemented this function for you
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.txt')]

    all_text = ''
    for f in files:
        all_text += ' '.join(open(directory + f).readlines()) + ' '

    return all_text

def normalize_log_prob(prob1, prob2):
    maxprob = max(prob1, prob2)

    prob1 -= maxprob
    prob2 -= maxprob
    prob1 = math.exp(prob1)
    prob2 = math.exp(prob2)

    normalize_constant = 1.0 / float(prob1 + prob2)
    prob1 *= normalize_constant
    prob2 *= normalize_constant

    return (prob1, prob2)

def calculate_doc_prob(training_model, testing_model, alpha):
    logprob = 0

    num_tokens_training = sum(training_model[1])
    num_words_training = len(training_model[0])

    for word in testing_model[0]:
        word_freq = testing_model[1][testing_model[0][word]]
        word_freq_in_training = 0
        if word in training_model[0]:
            word_freq_in_training = training_model[1][training_model[0][word]]
        for i in range(0, word_freq):
            logprob += math.log(word_freq_in_training + alpha)
            logprob -= math.log(num_tokens_training + num_words_training * alpha)

    return logprob

def create_BOW(sentence):
    bow_dict = {}
    bow = []

    sentence = sentence.lower()
    sentence = replace_non_alphabetic_chars_to_space(sentence)
    words = sentence.split(' ')
    for token in words:
        if len(token) < 1: continue
        if token not in bow_dict:
            new_idx = len(bow)
            bow.append(0)
            bow_dict[token] = new_idx
        bow[bow_dict[token]] += 1

    return bow_dict, bow

def replace_non_alphabetic_chars_to_space(sentence):
    return re.sub(r'[^a-z]+', ' ', sentence)

if __name__ == "__main__":
    main()