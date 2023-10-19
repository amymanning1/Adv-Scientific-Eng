import numpy as np

def func(x):
    f=np.square(x)-3
    return f

def bisection(lower_bound,upper_bound):
    f_lower = func(lower_bound)
    f_upper = func(upper_bound)


    if np.sign(f_lower) == np.sign(f_upper):
        raise Exception('A zero does not exist in these bounds')        

    while upper_bound-lower_bound >= 1e-15:    
        f_lower = func(lower_bound)
        f_upper = func(upper_bound)
        p=(lower_bound + upper_bound)/2
        f_new = func(p)
        if np.sign(f_new) == np.sign(f_upper):
            f_upper=f_new
            upper_bound=p
        elif np.sign(f_new) == np.sign(f_lower):
            f_lower=f_new
            lower_bound=p
    return p
def main():
    lower_bound=1
    upper_bound=2
    
    p = bisection(lower_bound,upper_bound)
        
    print('root = ')
    print(p)


if __name__ == '__main__':
    main()