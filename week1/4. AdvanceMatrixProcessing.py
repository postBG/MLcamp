import numpy

def main():
    A = get_matrix()
    print(matrix_tutorial(A))

def get_matrix():
    mat = []
    [n, m] = [int(x) for x in input().strip().split(" ")]
    for i in range(n):
        row = [int(x) for x in input().strip().split(" ")]
        mat.append(row)
    return numpy.array(mat)

def matrix_tutorial(A):
    # 2
    B = A.transpose()
    # 3
    try:
        C = numpy.linalg.inv(B)
    except:
        return "not invertible"
    # 4
    return numpy.sum(C > 0)

if __name__ == "__main__":
    main()