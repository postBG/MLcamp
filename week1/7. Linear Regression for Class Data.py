import statsmodels.api
import numpy

def main():
    (N, X, Y) = read_data()
    
    results = do_multivariate_regression(N, X, Y)
    print(results.summary())

    effective_variables = get_effective_variables(results)
    print(effective_variables)

def read_data():
    # 1
    f = open("students.dat", 'r')
    line = f.readline()
    
    N=0; X = []; Y = []
    
    while True:
        line = f.readline()
        if not line: break 
        splited = [float(x) for x in line.strip().split(" ")]
        X.append(splited[0:-1])
        Y.append(splited[-1])
        N = N + 1
        
    # X must be numpy.array in (30 * 5) shape.
    # Y must be 1-dimensional numpy.array.
    X = numpy.array(X)
    Y = numpy.array(Y)
    return (N, X, Y)

def do_multivariate_regression(N, X, Y):
    # 2
    results = statsmodels.api.OLS(Y, X).fit()
    return results

def get_effective_variables(results):
    eff_vars = []
	# 3
    p_vals = results.pvalues
    i=1
    for p_val in p_vals:
        if (p_val < 0.05):
            eff_vars.append('x%d' %i)
    return eff_vars

def print_students_data():
    with open("students.dat") as f:
        for line in f:
            print(line)

if __name__ == "__main__":
    main()