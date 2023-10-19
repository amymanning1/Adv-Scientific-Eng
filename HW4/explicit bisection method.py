import numpy as np

def func(x):
    f=np.square(x)-3
    return f

def main():
    lower_bound=1
    upper_bound=2
    f_lower = func(lower_bound)
    f_upper = func(upper_bound)

    while True:
        if np.sign(f_lower) == np.sign(f_upper):
            raise Exception('A zero does not exist in these bounds')
        
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
        
        if f_lower <= 1e-20:
            print('root = ')
            print(lower_bound)
            return False
        elif f_upper <= 1e-20:
            print('root = ')
            print(upper_bound)
            return False


if __name__ == '__main__':
    main()