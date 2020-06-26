T = int(input())
l1 = []
for i in range(T):
    l1.append(input())
count = []
for i in range(T):
    count.append(l1[i].count('apoc'))
for i in range(T):
    print(count[i])
