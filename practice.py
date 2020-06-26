from itertools import permutations

s = input()
string = list()
for word in s:
    string.append(word)

# print(string)
count = 0

combinations = permutations(string, 4)
# print(combinations)

distinct_combinations = list()
for i in list(combinations):
    # print(i)
    if i not in distinct_combinations:
        distinct_combinations.append(i)

print(distinct_combinations)
count = 0
for i in range(len(distinct_combinations)):
    temp = distinct_combinations[i]
    temp = ''.join(temp)
    #print(temp)
    #print(temp[::-1])
    if temp == temp[::-1]:
        count = count + 1
print(count)
