import re
import math

def main():
    # 1
    training_sentence = input()
    training_model = create_BOW(training_sentence)

    # 2
    testing_sentence = input()
    testing_model = create_BOW(testing_sentence)

    # 3
    alpha = float(input())

    print(calculate_doc_prob(training_model, testing_model, alpha))

def calculate_doc_prob(training_model, testing_model, alpha):
    # Implement likelihood function here...
    logprob = 0
    
    training_bow_dict = training_model[0]
    training_bow = training_model[1]
    d = len(training_bow_dict)
    N = sum(training_bow)
    
    testing_bow_dict = testing_model[0]
    testing_bow = testing_model[1]
    
    for word in testing_bow_dict.keys():
        if word in training_bow_dict:
            prob = (training_bow[training_bow_dict[word]] + alpha) / (N + (d * alpha))
        else:
            prob = alpha / (N + (d * alpha))
        logprob = logprob + (math.log(prob) * testing_bow[testing_bow_dict[word]])
        
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