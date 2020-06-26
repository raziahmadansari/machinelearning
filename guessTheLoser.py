def func(a, b, c):
    turn = 0
    k, l, n = a, b, c
    while True:
        if turn == 0:
            if (k + n) <= l:
                k = k + n
                n = n + 1
                turn = 1
            else:
                print('Nikhil')
                break
        else:
            if (k + n) <= l:
                k = k + n
                n = n + 1
                turn = 0
            else:
                print('Sahil')
                break



T = int(input())
l = list()
for i in range(T):
    #x = list(map(int, input().split()))
    x = [int(x) for x in input().split()]
    l.append(x)

for i in range(T):
    # print('inside loop')
    A = l[i][0]
    B = l[i][1]
    C = l[i][2]
    func(A, B, C)

