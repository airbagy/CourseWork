# (C) Jeff Orchard, 2021

import numpy as np



'''
=========================================

Var

=========================================
'''
class Var():
    '''
     x = Var(val)

     Creates an object of the Var class.

     Input:
       val   the value to be stored in the Var

     Output:
       x     object of Var class

     Then, we can get its value using any of:
       x.val
       x()

     The member x.creator is either None, or is a reference
     to the Operation object that created x.
     If x was creates by an Operation, then x.evaluate()
     will re-evaluate that Operation (and the subgraph below it).

     You can set the value using one of:
       x.val = 5
       x.set(5)

     The object also stores a derivative in x.grad. It is the
     derivative of the expression with respect to x.
    '''
    def __init__(self, val):
        self.val = val
        self.grad = None
        self.creator = None

    def set(self, val):
        self.val = val

    def evaluate(self):
        if self.creator!=None:
            self.val = self.creator.evaluate()
        return self.val

    def __repr__(self):
        if self.creator==None:
            return str(self.val)
        else:
            return self.creator.__repr__()

    def set_creator(self, op):
        self.creator = op

    def zero_grad(self):
        self.grad = 0.
        if self.creator!=None:
            self.creator.zero_grad()

    def backward(self, s=1.):
        self.grad += s
        if self.creator!=None:
            self.creator.backward(s)

    def __call__(self):
        return self.val

    def __add__(self, b):
        '''
         Wrapper so that 'x + b' is the same as 'x.plus(b)'
        '''
        return plus(self, b)

    def __sub__(self, b):
        '''
         Wrapper so that 'x - b' is the same as 'x.minus(b)'
        '''
        return minus(self, b)

    def __mul__(self, b):
        '''
         Wrapper so that 'x * b' is the same as 'x.mul(b)'
        '''
        return mul(self, b)

    def __truediv__(self, b):
        '''
         Wrapper so that 'x / b' is the same as 'x.div(b)'
        '''
        return div(self, b)

    def __pow__(self, b):
        '''
         Wrapper so that 'x**b' is the same as 'power(x, p=b)'
        '''
        return power(self, p=b)


'''
=========================================

 Wrapper Functions

=========================================
'''
def identity(a):
    '''
     y = identity(a)
     a is a Var
     y is a Var such that y.val = a.val
    '''
    c = Identity([a])()
    return c

def mul(a, b):
    '''
     y = a.mul(b)  or  y = a*b
     a and b are Vars
     y is a Var such that y.val = a.val * b.val
    '''
    c = Mul([a, b])()
    return c

def div(a, b):
    '''
     y = a.div(b)  or  y = a/b
     a and b are Vars
     y is a Var such that y.val = a.val / b.val
    '''
    c = Div([a, b])()
    return c

def plus(a, b):
    '''
     y = a.plus(b)  or  y = a+b
     a and b are Vars
     y is a Var such that y.val = a.val + b.val
    '''
    c = Plus([a, b])()
    return c

def minus(a, b):
    '''
     y = a.minus(b)  or  y = a-b
     a and b are Vars
     y is a Var such that y.val = a.val - b.val
    '''
    c = Minus([a, b])()
    return c

def recip(a):
    '''
     y = a.recip()
     a is a Var
     y is a Var such that y.val = 1. / a.val
    '''
    c = MyRecip([a])()
    return c

def power(a, p=2):
    '''
     y = a.power(p)  or  y = a**p
     a is a Var
     p is a scalar (not a Var)
     y is a Var such that y.val = a.val**p
    '''
    c = Power([a], p=p)()
    return c

def log(a):
    '''
     y = a.log()
     a is a Var
     y is a Var such that y.val = log(a.val)
    '''
    c = Log([a])()
    return c

def sin(a):
    '''
     y = a.sin()
     a is a Var
     y is a Var such that y.val = sin(a.val)
    '''
    c = Sin([a])()
    return c

def tan(a):
    '''
     y = a.tan()
     a is a Var
     y is a Var such that y.val = tan(a.val)
    '''
    c = Tan([a])()
    return c

def sqrt(a):
    '''
     y = a.sqrt()
     a is a Var
     y is a Var such that y.val = sqrt(a.val)
    '''
    c = Sqrt([a])()
    return c





'''
=========================================

 Operation

=========================================
'''
class Operation():
    '''
     op = Operation(args)

     Operation is an abstract base class for mathematical operations
     on scalars.

     Inputs:
       args  list of Var objects

     Output:
       op    a Operation object

     The Operation object op stores its arguments in the list op.args,
     and has the functions:
       op.__call__()
       op.__repr__()  # what to display for print(op)
       op.evaluate()
       op.zero_grad()
       op.backward()

     Usage:
       op()  # evaluates the operation without re-evaluating the args
       op.evaluate()  # re-evaluates the op after re-evaluating the args
       op.zero_grad() # resets grad to zero for all the args
       op.backward()  # propagates the derivative to the Vars below
    '''
    def __init__(self, args):
        self.args = args

    def __call__(self):
        raise NotImplementedError

    def evaluate(self):
        for a in self.args:
            a.evaluate()
        return self()()

    def zero_grad(self):
        for a in self.args:
            a.zero_grad()

    def backward(self, s=1.):
        raise NotImplementedError


'''
=========================================

 Operation Implementations

 Each type of mathematical operation is implemented
 by a different class, but all derived from the Operation
 base class.
=========================================
'''
class Identity(Operation):
    def __call__(self):
        y = Var(self.args[0].val)
        y.creator = self
        return y

    def __repr__(self):
        return self.args[0].__repr__()

    def backward(self, s=1.):
        self.args[0].backward(s)

class Log(Operation):
    def __call__(self):
        y = Var(np.log(self.args[0].val))
        y.creator = self
        return y

    def __repr__(self):
        return 'log('+self.args[0].__repr__()+')'

    def backward(self, s=1.):
        deriv = 1./self.args[0].val
        self.args[0].backward(s*deriv)


class Power(Operation):
    def __init__(self, args, p=2):
        super().__init__(args)
        self.p = p

    def __call__(self):
        y = Var(self.args[0].val**self.p)
        y.creator = self
        return y

    def __repr__(self):
        return '('+self.args[0].__repr__()+')**'+str(self.p)

    def backward(self, s=1.):
        deriv = self.p*self.args[0].val**(self.p-1)
        self.args[0].backward(s*deriv)


class Plus(Operation):
    def __call__(self):
        y = Var(self.args[0].val + self.args[1].val)
        y.creator = self
        return y

    def __repr__(self):
        return '('+self.args[0].__repr__()+'+'+str(self.args[1].__repr__())+')'

    def backward(self, s=1.):
        self.args[0].backward(s)
        self.args[1].backward(s)

class Minus(Operation):
    def __call__(self):
        y = Var(self.args[0].val - self.args[1].val)
        y.creator = self
        return y

    def __repr__(self):
        return '('+self.args[0].__repr__()+'-'+str(self.args[1].__repr__())+')'

    def backward(self, s=1.):
        self.args[0].backward(s)
        self.args[1].backward(-s)

class Recip(Operation):
    def __call__(self):
        y = Var(1./self.args[0].val)
        y.creator = self
        return y

    def __repr__(self):
        return '(1/'+self.args[0].__repr__()+')'

    def backward(self, s=1.):
        deriv = -1./self.args[0].val**2
        self.args[0].backward(s*deriv)


class Mul(Operation):
    def __call__(self):
        y = Var(self.args[0].val * self.args[1].val)
        y.creator = self
        return y

    def __repr__(self):
        return self.args[0].__repr__()+'*'+str(self.args[1].__repr__())

    def backward(self, s=1.):
        x_deriv = self.args[1].val
        y_deriv = self.args[0].val
        self.args[0].backward(s*x_deriv)
        self.args[1].backward(s*y_deriv)


class Div(Operation):
    def __call__(self):
        y = Var(self.args[0].val / self.args[1].val)
        y.creator = self
        return y

    def __repr__(self):
        return '('+self.args[0].__repr__()+')/('+str(self.args[1].__repr__()+')')

    def backward(self, s=1.):
        x_deriv = 1./self.args[1].val
        y_deriv = -self.args[0].val / self.args[1].val**2
        self.args[0].backward(s*x_deriv)
        self.args[1].backward(s*y_deriv)


class Sin(Operation):
    def __call__(self):
        y = Var(np.sin(self.args[0].val))
        y.creator = self
        return y

    def __repr__(self):
        return 'sin('+self.args[0].__repr__()+')'

    def backward(self, s=1.):
        deriv = np.cos(self.args[0].val)
        self.args[0].backward(s*deriv)

class Tan(Operation):
    def __call__(self):
        y = Var(np.tan(self.args[0].val))
        y.creator = self
        return y

    def __repr__(self):
        return 'tan('+self.args[0].__repr__()+')'

    def backward(self, s=1.):
        deriv = 1./np.cos(self.args[0].val)**2
        self.args[0].backward(s*deriv)

class Sqrt(Operation):
    def __call__(self):
        y = Var(np.sqrt(self.args[0].val))
        Y.creator = self
        return y

    def __repr__(self):
        return 'sqrt('+self.args[0].__repr__()+')'

    def backward(self, s=1.):
        deriv = 0.5/np.sqrt(self.args[0].val)
        self.args[0].backward(s*deriv)
