import numpy as np
import sys
import random


N = 6040
k = 3952


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def total_cost(C, ps1, ps2):
    return sum([cost(c, ps1, ps2) for c in C])


def cost(c, ps1, ps2):
    if len(c) == 1:
        m = c[0]
        return np.log(1 / ps1[m])

    s = 0
    for j, mj in enumerate(c):
        for t, mt in enumerate(c):
            if t < j:
                continue
            s += np.log(1 / ps2[j, t])

    return s / (len(c) - 1)


def probabilities(movies, ratings):
    m = len(movies)

    with open('data/movies per user.dat', 'r') as f:
        ns = np.loadtxt(f)

    with open('data/probabilities.dat', 'r') as f:
        ps1 = np.loadtxt(f)

    # joint probabilities only for the subset because of computing time
    ps2 = np.zeros((m, m))
    for j, mj in enumerate(movies):
        arrJ = np.where(ratings[:, j] > 0)[0]
        for t, mt in enumerate(movies):
            arrT = np.where(ratings[:, t] > 0)[0]
            arr = np.intersect1d(arrJ, arrT)
            ps2[j, t] = sum([ratings[i, j] * ratings[i, t] * 2 / (ns[i] * (ns[i] - 1)) for i in arr])
    ps2 = (ps2 + 2 / (k * (k - 1))) / (N + 1)

    return ps1, ps2


def correlations(movies, ps1, ps2):
    m = len(movies)

    cs = np.zeros((k, k))
    cs2 = np.zeros((m, m))
    for j, mj in enumerate(movies):
        for t, mt in enumerate(movies):
            cs[mj, mt] = (ps2[j, t] >= ps1[j] * ps1[t])
            cs2[j, t] = (ps2[j, t] >= ps1[j] * ps1[t])

    return cs


def ccpivot(movies, cs):
    if len(movies) == 0:
        return []

    i = random.choice(movies)
    c = [i]
    v2 = []

    for j in movies:
        if j == i:
            continue

        if cs[i, j]:
            c.append(j)
        else:
            v2.append(j)

    return [c] + ccpivot(v2, cs)


def after_pivot(C, ps1, ps2):
    after_C = C
    length = len(after_C)
    j = 0
    while j < 5:
        i = 0
        for c1 in after_C:
            cmin = -1
            min_cost = -1
            for c2 in after_C:
                if c1 == c2:
                    continue
                double_cost = cost(c1 + c2, ps1, ps2)
                single_cost = total_cost([c1, c2], ps1, ps2)
                if double_cost < single_cost:
                    if cmin == -1:
                        cmin = c2
                        min_cost = double_cost
                    elif double_cost < min_cost:
                        cmin = c2
                        min_cost = double_cost
            if cmin != -1:
                after_C.remove(c1)
                after_C.remove(cmin)
                after_C += [c1 + cmin]
            else:
                i += 1
            if i == length:
                # print('Ohhh')
                break
        j += 1
    # i = 0
    # for c1 in after_C:
    #     cmin = -1
    #     min_cost = -1
    #     for c2 in after_C:
    #         if c1 == c2:
    #             continue
    #         double_cost = cost(c1 + c2, ps1, ps2)
    #         single_cost = total_cost([c1, c2], ps1, ps2)
    #         if double_cost < single_cost:
    #             if cmin == -1:
    #                 cmin = c2
    #                 min_cost = double_cost
    #             elif double_cost < min_cost:
    #                 cmin = c2
    #                 min_cost = double_cost
    #     if cmin != -1:
    #         after_C.remove(c1)
    #         after_C.remove(cmin)
    #         after_C += [c1 + cmin]
    #     else:
    #         i += 1
    #     if i == length:
    #         # print('Ohhh')
    #         break
    # print(total_cost(after_C, ps1, ps2))
    # for c1 in after_C:
    #     if len(c1) == 1:
    #         for c2 in after_C:
    #             if c1 == c2:
    #                 continue
    #
    #             if cost(c1 + c2, ps1, ps2) < total_cost([c1, c2], ps1, ps2):
    #                 after_C.remove(c1)
    #                 after_C.remove(c2)
    #                 after_C += [c1 + c2]
    #                 break
    return after_C


def main():
    file = sys.argv[3]
    alg = int(sys.argv[2])
    if alg != 1 and alg != 2:
        eprint('bad algorithm dawg')
        exit()

    movies = sorted(random.sample(range(3952), 100))
    # movies = [36, 48, 177, 183, 240, 242, 245, 253, 271, 355, 368, 378, 452, 455, 526, 660, 665, 705, 837, 850, 952,
    #           974, 1026, 1027, 1031, 1035, 1075, 1081, 1113, 1118, 1175, 1194, 1199, 1205, 1248, 1336, 1348, 1412, 1455,
    #           1545, 1554, 1581, 1759, 1776, 1810, 1821, 1833, 1931, 2040, 2057, 2105, 2126, 2151, 2184, 2193, 2222,
    #           2274, 2318, 2334, 2372, 2412, 2425, 2426, 2532, 2537, 2577, 2579, 2598, 2616, 2650, 2669, 2674, 2689,
    #           2713, 2846, 2855, 2913, 2922, 3008, 3121, 3187, 3234, 3257, 3308, 3322, 3341, 3393, 3396, 3417, 3436,
    #           3504, 3524, 3597, 3618, 3623, 3676, 3745, 3860, 3906, 3917]
    # with open('randomsubset1.txt', 'w') as f:
    #     np.savetxt(f, np.array(movies) + 1, fmt='%i')
    with open(file, 'r') as f:
        movies = list(np.loadtxt(f).astype(int) - 1)

    for m in movies:
        if (type(m) != np.int32 and type(m) != int) or m > 3952 or m < 1:
            eprint('bad movies dawg')
            exit()
    print(movies)
    with open('data/ratings.dat', 'r') as f:
        ratings = np.loadtxt(f, usecols=movies)

    to_remove = []
    for j, mj in enumerate(movies):
        num = sum(ratings[:, j])
        if num < 10:
            eprint(f'Movie {mj} ignored because it has only {num} ratings')
            to_remove += [mj]

    movies = [m-1 for m in movies if m not in to_remove]

    with open('data/ratings.dat', 'r') as f:
        ratings = np.loadtxt(f, usecols=movies)
    print(movies)

    ps1, ps2 = probabilities(movies, ratings)
    cs = correlations(movies, ps1, ps2)
    C = ccpivot(movies, cs)
    print(total_cost(C, ps1, ps2))
    if alg == 2:
        tmp = C
        C = after_pivot(C, ps1, ps2)
    with open('titles.txt', 'r') as f:
        titles = np.loadtxt(f, delimiter='\t', dtype=str)
    for c in C:
        for m in c:
            print(f'{m+1} "{titles[m]}"', end=', ')
        print()
    print(total_cost(C, ps1, ps2))
    # print(sorted([item for sublist in C for item in sublist]) == sorted([item for sublist in tmp for item in sublist]))


if __name__ == '__main__':
    main()
