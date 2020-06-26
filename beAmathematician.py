N = int(input())
X, Y = map(int, input().split())

x = 0
y = 0
for i in range(N + 1):
    if i % X == 0 and i % Y == 0:
        continue
    elif i % X == 0:
        x = x + 1
    elif i % Y == 0:
        y = y + 1

if x > y:
    print("ABHISHEK", (x - y))
elif y > x:
    print("PARTH", (y - x))
else:
    print("TIE")
