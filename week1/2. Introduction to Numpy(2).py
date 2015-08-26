import numpy

def main():
    print(matrix_tutorial())

def matrix_tutorial():
    A = numpy.array([[1,4,5,8], [2,1,7,3], [5,4,5,9]])

    # 1
    B = A.reshape((6,2))
    # 2
    x = numpy.array([[2,2],[5,3]])
    l = numpy.concatenate((B,x),axis = 0)
    # 3
    s = numpy.split(l,2,axis = 0)
    C = s[0]
    D = s[1]
    # 4
    E = numpy.concatenate((C,D),axis = 1)
    # 5
    return E

if __name__ == "__main__":
    main()