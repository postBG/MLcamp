import numpy
import elice_utils

def main():
    num_flips = int(input())
    prob_head = float(input())

    coin_results = flip_multiple_times(num_flips, prob_head)
    print(visualize(coin_results))

def flip_a_coin(prob_head):
    random_num = numpy.random.random()

    # exercise
    if random_num < prob_head:
        return "head"
    else:
        return "tail"

def flip_multiple_times(num_flips, prob_head):
    if num_flips > 100000:
        print("[num_flips] should be less than 100000")
        num_flips = 100000

    coin_results = []
    for i in range(0, num_flips):
        coin_results.append(flip_a_coin(prob_head))

    return coin_results

def visualize(coin_results):
    num_heads = coin_results.count("head")
    num_tails = coin_results.count("tail")
    heads_percentage = num_heads / (num_heads + num_tails)
    tails_percentage = num_tails / (num_heads + num_tails)
    return elice_utils.visualize_boxplot("Coin Flip: %d times" % (num_heads + num_tails),
        [heads_percentage, tails_percentage],
        ["heads (%)", "tails (%)"])

if __name__ == "__main__":
    main()