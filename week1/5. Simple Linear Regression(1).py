import numpy

def main():
    (N, X, Y) = read_data()
    print(N)
    print(X)
    print(Y)

def read_data():
    # 1
    N = int(input())
    # 2
    X = []
    Y = []
    for i in range(N):
        [x, y] = [float(x) for x in input().strip().split(" ")]
        X.append(x)
        Y.append(y)
        
    return (N, X, Y)

if __name__ == "__main__":
    main()