import itertools

def find_gcd(x, y):
    while (y):
        x, y = y, x % y

    return x


def findsubsets(s, n):
    return list(itertools.combinations(s, n))



# Driver Code
# l = [2, 4, 6, 8, 16]
T = int(input())
l1 = list()
l2 = list()
for _ in range(T):
    x1 = list(map(int, input().split()))
    x = list()

    x = list(map(int, input().split()))
    l1.append(x)
    l2.append(x1)


'''print(l1)
print(l2)'''


for i in range(len(l1)):
    num1 = l1[i][0]
    num2 = l1[i][1]
    gcd_default = find_gcd(num1, num2)

    for j in range(2, len(l1[i])):
        gcd_default = find_gcd(gcd_default, l1[i][j])

    print(gcd_default, end=' ')

    '''subsets = findsubsets(l1[i],(len(l1[i]) - l2[i][1]))
    print(subsets)

    for k in range(len(subsets)):'''

    N = l2[i][0]
    D = l2[i][1]
    minimum = min(l1[i])
    max_gcd = gcd_default + 1
    temp_list = l1[i]
    for k in range(len(temp_list)):
        for l in range()













'''num1 = l[0]
num2 = l[1]
gcd = find_gcd(num1, num2)

for i in range(2, len(l)):
    gcd = find_gcd(gcd, l[i])

print(gcd)'''