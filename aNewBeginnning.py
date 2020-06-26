T = int(input())
l1 = []
for _ in range(T):
    l1.append(input())

dir = 1
X = 0
Y = 0

for i in range(T):
    X = 0
    Y = 0
    point = []
    temp = []
    sample = l1[i]

    for j in range(len(sample)):
        if sample[j] == 'F':
            temp.append(sample[:j] + 'R' + sample[j+1:])
            temp.append(sample[:j] + 'L' + sample[j+1:])
        elif sample[j] == 'L':
            temp.append(sample[:j] + 'F' + sample[j + 1:])
            temp.append(sample[:j] + 'R' + sample[j + 1:])
        elif sample[j] == 'R':
            temp.append(sample[:j] + 'F' + sample[j + 1:])
            temp.append(sample[:j] + 'L' + sample[j + 1:])

    #print(temp)
    for i in range(len(temp)):
        sample = temp[i]
        dir = 1
        X = 0
        Y = 0
        for j in range(len(sample)):
            if sample[j] == 'F':
                if dir == 1:
                    Y = Y + 1
                elif dir == 3:
                    Y = Y - 1
                elif dir == 0:
                    X = X + 1
                elif dir == 2:
                    X = X - 1

            elif sample[j] == 'L':
                dir = (dir + 1) % 4
            elif sample[j] == 'R':
                dir = (dir - 1) % 4

        point.append(str(X) + ',' + str(Y))

    count = 0
    distinct = []
    for i in range(len(point)):
        if point[i] not in distinct:
            distinct.append(point[i])
    print(len(distinct))
    #print(distinct)
