#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <unordered_set>
#include <algorithm>

using namespace std;
class Value {
public:
    double data;
    double grad;
    function<void()> _backward;
    vector<Value*> _prev;
    string _op;
    string label;

    // Constructor
    Value(double data_val, const vector<Value*>& children = {}, 
          const string& op = "", const string& label = "") 
        : data(data_val), grad(0.0), _op(op), label(label) 
    {
        // Add all children to _prev
        for (Value* child : children) {
            _prev.push_back(child);
        }
        
        // Default backward function does nothing
        _backward = [](){};
    }
    
    // For printing the value
    void print() const {
        cout << "Value(data=" << data << ")" << endl;
    }
    
    // Addition operator
    Value operator+(Value& other) {
        // For non-Value types, we'd need more code, but let's keep it simple
        Value out(this->data + other.data, {this, &other}, "+");
        
        // Define the backward function
        out._backward = [this, &other, &out]() {
            this->grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };
        
        return out;
    }
    
    // Multiplication operator
    Value operator*(Value& other) {
        Value out(this->data * other.data, {this, &other}, "*");
        
        // Define the backward function
        out._backward = [this, &other, &out]() {
            this->grad += other.data * out.grad;
            other.grad += this->data * out.grad;
        };
        
        return out;
    }
    
    // Power operator overload
    Value operator^(double exponent) {
        Value out(std::pow(this->data, exponent), {this}, "**" + to_string(exponent));
        
        // Define the backward function
        out._backward = [this, exponent, &out]() {
            this->grad += exponent * std::pow(this->data, exponent - 1) * out.grad;
        };
        
        return out;
    }
    
    // Scalar multiplication (for cases like 2.0 * v, implemented with friend)
    friend Value operator*(double scalar, Value& v) {
        Value scalar_value(scalar);
        return v * scalar_value;
    }
    
    // Division operator (self / other)
    Value operator/(Value& other) {
        // Division is self * other^(-1)
        Value inv = other ^ -1.0;
        return *this * inv;
    }
    
    // Negation operator (-self)
    Value operator-() {
        // Negation is -1 * self
        Value neg_one(-1.0);
        return neg_one * *this;
    }
    
    // Subtraction operator (self - other)
    Value operator-(Value& other) {
        // Subtraction is self + (-other)
        Value neg_other = -other;
        return *this + neg_other;
    }
    
    // Reverse addition operator (for cases like 2.0 + v, equivalent to Python's __radd__)
    friend Value operator+(double scalar, Value& v) {
        Value scalar_value(scalar);
        return v + scalar_value;
    }
    
    // Hyperbolic tangent (tanh) activation function
    Value tanh() {
        double x = this->data;
        double t = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);
        Value out(t, {this}, "tanh");
        
        // Define the backward function
        out._backward = [this, t, &out]() {
            this->grad += (1 - t*t) * out.grad;
        };
        
        return out;
    }
    
    // Exponential function (e^x)
    Value exp() {
        double x = this->data;
        double result = std::exp(x);
        Value out(result, {this}, "exp");
        
        // Define the backward function
        // The derivative of e^x is e^x itself, which is out.data
        out._backward = [this, &out]() {
            this->grad += out.data * out.grad;
        };
        
        return out;
    }
    
    // Backward method to perform backpropagation through the entire computational graph
    void backward() {
        // Stores nodes in topological order
        vector<Value*> topo;
        unordered_set<Value*> visited;
        
        // Helper function to build topological ordering (recursive DFS)
        function<void(Value*)> build_topo = [&](Value* v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (Value* child : v->_prev) {
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };
        
        // Build the topological ordering starting from this node
        build_topo(this);
        
        // Initialize gradient of output to 1.0
        this->grad = 1.0;
        
        // Backpropagate in reverse topological order
        reverse(topo.begin(), topo.end());
        for (Value* node : topo) {
            node->_backward();
        }
    }
};

int main() {
    
    
    return 0;
}