def a(x):
    x = x / 30
        
    # 4차
    coeff = [0.0243987, 0.49096448, 1.08571579, 0.01212056, -0.69068458]
    
    # 8차        
    # coeff = [0.0172036, 0.49211715, 1.90813097, 0.0811299, -5.34212661, -0.16520139, 7.32628553, 0.09354028, -3.44125495]
    
    result = 0
    for i in range(len(coeff)):
        result += coeff[i] * x ** i
    
    result *= 30
    return result


def b(x):
    coeff = [0.7319608837547098, 
             0.4909644773184077, 
             0.036190526341102945, 
             1.3467289813403963e-05,
             -2.5580910291249533e-05,]
    e = coeff[0]
    d = coeff[1]
    c = coeff[2]
    b = coeff[3]
    a = coeff[4]
    
    e, d, c, b = e/a, d/a, c/a, b/a
    print(a, b, c, d, e)
    
    return a*((x*x)*(x*(x+b)+c)+d*x+e)    
    
x = -0.5 * 30

print(a(x))
print(b(x))