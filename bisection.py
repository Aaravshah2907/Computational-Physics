from math import e, log10
import matplotlib.pyplot as plt

tolerance = 1e-10

def bisection(f, a, b, tol=tolerance):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    
    errors = []
    iterations = 0
    prev_midpoint = None
    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        iterations += 1
        if prev_midpoint is not None:
            errors.append(abs(midpoint - prev_midpoint))
        if f(midpoint) == 0:
            break
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        prev_midpoint = midpoint
    # Final error (between last two midpoints)
    if prev_midpoint is not None and abs(midpoint - prev_midpoint):
        final_midpoint = (a + b) / 2.0
        errors.append(abs(final_midpoint - prev_midpoint))
    return (a + b) / 2.0, errors

def Newton_Raphson(f, df, x0, tol=tolerance):
    x = x0
    errors = []
    iterations = 0
    prev_x = None
    while abs(f(x)) > tol:
        iterations += 1
        if df(x) == 0:
            raise ValueError("Derivative is zero. No solution found.")
        x_new = x - f(x) / df(x)
        if prev_x is not None:
            if abs(x_new - prev_x):
                errors.append(abs(x_new - prev_x))
        prev_x = x_new
        x = x_new
    # Final error (between last two approximations)
    if prev_x is not None and abs(x - prev_x):
        errors.append(abs(x - prev_x))
    return x, errors

def df(x):
    return -e**-x - 1

def f(x):
    return e**-x - x

#%%
def secant_method(f, x0, x1, tol=10e-10):
    errors = []
    iterations = 0
    prev_x = None
    x_new = 0
    while abs(f(x1)) > tol:
        iterations += 1
        if f(x1) == f(x0):
            raise ValueError("Function values at x0 and x1 are equal. No solution found.")
        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if prev_x is not None:
            if abs(x_new - prev_x):
                errors.append(abs(x_new - prev_x))
        prev_x = x_new
        x0, x1 = x1, x_new
    # Final error (between last two approximations)
    if prev_x is not None and abs(x_new - prev_x):
        errors.append(abs(x_new - prev_x))
    return x_new, errors

if __name__ == "__main__":
    a = float(input("Enter lower bound (a): "))
    b = float(input("Enter upper bound (b): "))

    try:
        root, bisect_errors = bisection(f, a, b, tol=tolerance)
        print(f"Bisection : Root found: {root}")
    except ValueError as e:
        print(e)
        bisect_errors = []

    try:
        x0 = float(input("Enter initial guess for Newton-Raphson method: "))
        root_newton, nr_errors = Newton_Raphson(f, df, x0, tol=tolerance)
        print(f"Newton-Raphson: Root found: {root_newton}")
    except ValueError as e:
        print(e)
        nr_errors = []

    try:
        x0_secant = float(input("Enter first initial guess for Secant method: "))
        x1_secant = float(input("Enter second initial guess for Secant method: "))
        root_secant, secant_errors = secant_method(f, x0_secant, x1_secant, tol=tolerance)
        print(f"Secant Method: Root found: {root_secant}")
    except ValueError as e:
        print(e)
        secant_errors = []

    # Plotting
    plt.figure()
    if bisect_errors:
        plt.plot(range(1, len(bisect_errors)+1), [log10(err) for err in bisect_errors], label="Bisection")
    if nr_errors:
        plt.plot(range(1, len(nr_errors)+1), [log10(err) for err in nr_errors], label="Newton-Raphson")
    if secant_errors:
        plt.plot(range(1, len(secant_errors)+1), [log10(err) for err in secant_errors], label="Secant Method")
    plt.xlabel("Iteration")
    plt.ylabel("log10(Error)")
    plt.title("log10(Error) vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()
