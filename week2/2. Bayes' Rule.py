def main():
    sensitivity = float(input())
    prior_prob = float(input())
    false_alarm = float(input())

    print("%.2lf%%" % (100 * mammogram_test(sensitivity, prior_prob, false_alarm)))

def mammogram_test(sensitivity, prior_prob, false_alarm):
    p_a1_b1 = sensitivity # p(A = 1 | B = 1)

    p_b1 = prior_prob # p(B = 1)

    p_b0 = 1 - prior_prob # p(B = 0)

    p_a1_b0 = false_alarm # p(A = 1|B = 0)

    p_a1 = (p_a1_b1 * p_b1) + (p_a1_b0 * p_b0) # p(A = 1)

    p_y1_x1 = (p_a1_b1 * p_b1) / p_a1 # p(B = 1|A = 1)

    return p_y1_x1

if __name__ == "__main__":
    main()