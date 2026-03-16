from collections import Counter

def choose_best_answer(answers):

    counter = Counter(answers)

    best_answer = counter.most_common(1)[0][0]

    return best_answer
