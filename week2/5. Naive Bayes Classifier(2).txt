import re
import math

def main():
    training1_sentence = input()
    training2_sentence = input()
    testing_sentence = input()

    alpha = float(input())
    prob1 = float(input())
    prob2 = float(input())

    print(naive_bayes(training1_sentence, training2_sentence, testing_sentence, alpha, prob1, prob2))

def naive_bayes(training1_sentence, training2_sentence, testing_sentence, alpha, prob1, prob2):
    # Implement Naive Bayes Algorithm here...
    training_model1 = create_BOW(training1_sentence)
    training_model2 = create_BOW(training2_sentence)
    testing_model = create_BOW(testing_sentence)
    
    likelihood1 = calculate_doc_prob(training_model1, testing_model, alpha)
    likelihood2 = calculate_doc_prob(training_model2, testing_model, alpha)
    
    classify1 = math.log(prob1) + likelihood1
    classify2 = math.log(prob2) + likelihood2
    
    return normalize_log_prob(classify1, classify2)

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