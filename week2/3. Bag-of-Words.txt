import re

def main():
    sentence = input()
    BOW_dict, BOW = create_BOW(sentence)

    print(BOW_dict)
    print(BOW)

def create_BOW(sentence):
    # Exercise
    sentence = sentence.lower()
    sentence = replace_non_alphabetic_chars_to_space(sentence)
    words = [x.lower() for x in sentence.strip().split(" ")]
    bow_dict = {}
    bow = []
    count = 0
    
    for word in words:
        if len(word) < 1: continue
        if word not in bow_dict:
            bow_dict[word] = count
            bow.append(1)
            count = count + 1
        else:
            bow[bow_dict[word]] = bow[bow_dict[word]] + 1
            
    return bow_dict, bow

def replace_non_alphabetic_chars_to_space(sentence):
    return re.sub(r'[^a-z]+', ' ', sentence)

if __name__ == "__main__":
    main()