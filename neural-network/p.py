a = [1, 5, 6, 7]
b = [2, 4, 5, 8, 10]

def sort(a, b):
    c = []
    x = 0
    y = 0
    while x < len(a) and y < len(b) :
        if a[x] <= b[y]:
            c.append(a[x])
            x += 1
        else:
            c.append(b[y])
            y += 1
    a = a[x::]
    b = b[y::]
    c += a
    c += b
    return c
print(sort(a, b))
print(len(a))

