import sys
# sys.path.append("./")

import math
from .utils import need_grad

class Val:
    def __init__(self, data, _child = (), require_grad: bool = True, dtype=float):
        self.data = dtype(data)
        self.dtype = dtype
        self.grad = 0.0
        self._prev = set(_child)
        self.require_grad = require_grad
        self._backward = lambda : None
    

    def __str__(self):
        return str(self.data)

    def __add__(self, other):
        if not isinstance(other, Val):
            other = Val(other, require_grad=False)
        out = Val(self.data + other.data, (self, other))
        
        @need_grad(self)
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        
        return out

    
    def __sub__(self, other):
        if not isinstance(other, Val):
            other = Val(other, require_grad=False)
        out = Val(self.data - other.data, (self, other))

        @need_grad(self)
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad -= 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        if not isinstance(other, Val):
            other = Val(other, require_grad=False)
        out = Val(self.data * other.data, (self, other))

        @need_grad(self)
        def _backward():
            other.grad += self.data * out.grad
            self.grad += other.data * out.grad
        
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        if not isinstance(other, Val):
            other = Val(other, require_grad=False)
        assert other.data != 0, "Division by zero"
        
        out = Val(self.data / other.data, (self, other))
        
        @need_grad(self)
        def _backward():
            self.grad += out.grad * (1/other.data)
            other.grad += -out.grad * self.data * (1/other.data**2)
        
        out._backward = _backward
        
        return out
    
    def __pow__(self, power, ):
        out = Val(self.data**power, (self,))
        
        @need_grad(self)
        def _backward():
            self.grad += power * out.data / self.data * out.grad
            
        out._backward = _backward
        return out
        
    
    def exp(self):
        out = Val(math.exp(self.data), (self,))
        
        @need_grad(self)
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def ln(self, a: float =1, b: float =0):
        out = Val(math.log(self.data), (self,))
        
        @need_grad(self)
        def _backward():
            self.grad += a * out.data * out.grad
        
        out._backward = _backward
        return out
    
    def tanh(self):
        mid = math.exp(2*self.data)
        out = Val((mid - 1.0)/(mid + 1.0), (self,))
        
        @need_grad(self)
        def _backward():
            self.grad += out.grad * (1 - out.data**2)

        out._backward = _backward
    
        return out
    
    def backward(self):
         # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)

