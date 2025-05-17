#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <memory> // For shared_ptr

#include <set> // For initial set of children, similar to Python

using namespace std;

// Forward declaration for friend operators
class Value;

// Using shared_ptr for managing Value objects in the graph
// Inherit from enable_shared_from_this to get a shared_ptr to *this
class Value : public enable_shared_from_this<Value> {
public:
    double data;
    double grad;
    function<void()> _backward;
    vector<shared_ptr<Value>> _prev; // Use shared_ptr to parents
    string _op;
    string label;

    // Constructor - takes a set of shared_ptrs for children
    Value(double data_val, const set<shared_ptr<Value>>& children = {},
          const string& op = "", const string& label = "")
        : data(data_val), grad(0.0), _op(op), label(label)
    {
        // Copy shared_ptrs from the set to the vector
        _prev.assign(children.begin(), children.end());

        // Default backward function does nothing
        _backward = [](){};
    }

    // Constructor - also allow vector of shared_ptrs for convenience
     Value(double data_val, const vector<shared_ptr<Value>>& children_vec,
          const string& op = "", const string& label = "")
        : data(data_val), grad(0.0), _op(op), label(label)
    {
        _prev = children_vec;
        _backward = [](){};
    }


    // For printing the value using << operator
    friend ostream& operator<<(ostream& os, const Value& v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad << ", label=" << v.label << ")";
        return os;
    }

    // --- Operator Overloads ---

    // Addition operator (Value + Value)
    shared_ptr<Value> operator+(const shared_ptr<Value>& other) const {
        // Create new Value object on the heap using make_shared
        shared_ptr<Value> out = make_shared<Value>(this->data + other->data, vector<shared_ptr<Value>>{const_cast<Value*>(this)->shared_from_this(), other}, "+");

        // Define the backward function
        // Capture shared_ptr's by value to keep parents and output node alive
        out->_backward = [self = const_cast<Value*>(this)->shared_from_this(), other, out]() {
            self->grad += 1.0 * out->grad;
            other->grad += 1.0 * out->grad;
        };

        return out;
    }

    // Addition operator (Value + double)
    shared_ptr<Value> operator+(double scalar) const {
        // Create a temporary Value for the scalar and use Value + Value operator
        shared_ptr<Value> scalar_value = make_shared<Value>(scalar);
        return *this + scalar_value;
    }

    // Multiplication operator (Value * Value)
    shared_ptr<Value> operator*(const shared_ptr<Value>& other) const {
        shared_ptr<Value> out = make_shared<Value>(this->data * other->data, vector<shared_ptr<Value>>{const_cast<Value*>(this)->shared_from_this(), other}, "*");

        out->_backward = [self = const_cast<Value*>(this)->shared_from_this(), other, out]() {
            self->grad += other->data * out->grad;
            other->grad += self->data * out->grad;
        };

        return out;
    }

    // Multiplication operator (Value * double)
     shared_ptr<Value> operator*(double scalar) const {
        shared_ptr<Value> scalar_value = make_shared<Value>(scalar);
        return *this * scalar_value;
    }


    // Power operator overload (Value ^ double)
    shared_ptr<Value> operator^(double exponent) const {
        if (fmod(exponent, 1.0) != 0.0 && this->data < 0) {
             throw runtime_error("Cannot take fractional power of negative number");
        }
        shared_ptr<Value> out = make_shared<Value>(std::pow(this->data, exponent), vector<shared_ptr<Value>>{const_cast<Value*>(this)->shared_from_this()}, "**" + to_string(exponent));

        out->_backward = [self = const_cast<Value*>(this)->shared_from_this(), exponent, out]() {
             // d(x^n)/dx = n * x^(n-1)
            self->grad += exponent * std::pow(self->data, exponent - 1) * out->grad;
        };

        return out;
    }

    // Division operator (Value / Value) --> Implemented as Value * (Other ^ -1) in Python
    shared_ptr<Value> operator/(const shared_ptr<Value>& other) const {
         return (*this) * ((*other) ^ -1.0); // Use power and multiplication operators
    }

    // Division operator (Value / double)
     shared_ptr<Value> operator/(double scalar) const {
         return (*this) * (make_shared<Value>(scalar) ^ -1.0); // Use power and multiplication operators
     }

    // Negation operator (-Value) --> Implemented as Value * -1 in Python
    shared_ptr<Value> operator-() const {
        return (*this) * -1.0; // Use Value * double operator
    }

    // Subtraction operator (Value - Value) --> Implemented as Value + (-Other) in Python
    shared_ptr<Value> operator-(const shared_ptr<Value>& other) const {
        return (*this) + (-other); // Use Value + Value and unary - operators
    }

    // Subtraction operator (Value - double)
    shared_ptr<Value> operator-(double scalar) const {
        shared_ptr<Value> scalar_value = make_shared<Value>(scalar);
        return (*this) - scalar_value; // Use Value - Value operator
    }


    // --- Friend Operators (for scalar on the left side) ---
    // Needs to be friends to access the private constructor or just call the existing operators

    friend shared_ptr<Value> operator*(double scalar, const shared_ptr<Value>& v) {
        // Create a temporary Value for the scalar and use Value * Value operator
        shared_ptr<Value> scalar_value = make_shared<Value>(scalar);
        return scalar_value * v;
    }

    friend shared_ptr<Value> operator+(double scalar, const shared_ptr<Value>& v) {
        // Create a temporary Value for the scalar and use Value + Value operator
        shared_ptr<Value> scalar_value = make_shared<Value>(scalar);
        return scalar_value + v;
    }

     friend shared_ptr<Value> operator-(double scalar, const shared_ptr<Value>& v) {
        // Implement as scalar + (-v)
        shared_ptr<Value> scalar_value = make_shared<Value>(scalar);
        return scalar_value + (-v);
    }

     friend shared_ptr<Value> operator/(double scalar, const shared_ptr<Value>& v) {
        // Implement as scalar * (v ^ -1)
         shared_ptr<Value> scalar_value = make_shared<Value>(scalar);
         return scalar_value * ((*v) ^ -1.0);
    }


    // --- Activation Functions ---

    shared_ptr<Value> tanh() const {
        double x = this->data;
        double t = (std::exp(2*x) - 1)/(std::exp(2*x) + 1);
        shared_ptr<Value> out = make_shared<Value>(t, vector<shared_ptr<Value>>{const_cast<Value*>(this)->shared_from_this()}, "tanh");

        out->_backward = [self = const_cast<Value*>(this)->shared_from_this(), t, out]() {
            // d(tanh(x))/dx = 1 - tanh(x)^2
            self->grad += (1 - t*t) * out->grad;
        };

        return out;
    }

    shared_ptr<Value> exp() const {
        double x = this->data;
        double result = std::exp(x);
        shared_ptr<Value> out = make_shared<Value>(result, vector<shared_ptr<Value>>{const_cast<Value*>(this)->shared_from_this()}, "exp");

        out->_backward = [self = const_cast<Value*>(this)->shared_from_this(), out]() {
            // d(e^x)/dx = e^x
            self->grad += out->data * out->grad; // Derivative is the output itself
        };

        return out;
    }


    // --- Backward Pass ---

    void backward() {

        // Build topological order of the graph
        vector<shared_ptr<Value>> topo;
        unordered_set<shared_ptr<Value>> visited; // Use unordered_set for efficient lookup

        // Recursive DFS to build topological sort
        function<void(shared_ptr<Value>)> build_topo =
            [&](shared_ptr<Value> v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (const auto& child : v->_prev) {
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };

        // Start topological sort from 'this' node
        build_topo(shared_from_this());

        // Initialize the gradient of the output node
        this->grad = 1.0;

        // Backpropagate gradients in reverse topological order
        reverse(topo.begin(), topo.end());
        for (const auto& node : topo) {
            node->_backward(); // Execute the stored backward function for each node
        }
    }
};


// Main function for testing
int main() {
    // Example usage (similar to micrograd examples)

    // Create Value objects using make_shared
    shared_ptr<Value> a = make_shared<Value>(2.0, set<shared_ptr<Value>>{}, "", "a");
    shared_ptr<Value> b = make_shared<Value>(-3.0, set<shared_ptr<Value>>{}, "", "b");
    shared_ptr<Value> c = make_shared<Value>(10.0, set<shared_ptr<Value>>{}, "", "c");

    // Build computational graph using overloaded operators
    shared_ptr<Value> d = (*a) * b; d->label = "d"; // Dereference shared_ptr to call operator*
    shared_ptr<Value> e = (*d) + c; e->label = "e";
    shared_ptr<Value> f = e->tanh(); f->label = "f";

    cout << "Forward pass complete." << endl;
    cout << "f: " << *f << endl; // Use the overloaded << operator

    cout << "\nBackward pass..." << endl;
    f->backward(); // Perform backpropagation
    cout << "Backward pass complete." << endl;

    // Print gradients (access members via -> operator for shared_ptr)
    cout << "a: " << *a << endl;
    cout << "b: " << *b << endl;
    cout << "c: " << *c << endl;
    cout << "d: " << *d << endl;
    cout << "e: " << *e << endl;
    cout << "f: " << *f << endl; // grad should be 1.0 for the output node

    // Test scalar operations
    cout << "\nTesting scalar operations:" << endl;
    shared_ptr<Value> x = make_shared<Value>(5.0, set<shared_ptr<Value>>{}, "", "x");
    shared_ptr<Value> y = x + 2.0; y->label = "y";
    shared_ptr<Value> z = 3.0 * y; z->label = "z";
    shared_ptr<Value> w = z - 1.0; w->label = "w";
    shared_ptr<Value> v = w / 2.0; v->label = "v";
    shared_ptr<Value> u = 10.0 / v; u->label = "u";

    cout << "Forward pass (scalars) complete." << endl;
    cout << "u: " << *u << endl;

    cout << "\nBackward pass (scalars)..." << endl;
    u->backward();
    cout << "Backward pass (scalars) complete." << endl;

    cout << "x: " << *x << endl;
    cout << "y: " << *y << endl;
    cout << "z: " << *z << endl;
    cout << "w: " << *w << endl;
    cout << "v: " << *v << endl;
    cout << "u: " << *u << endl;


    // More complex expression (mimicking a neuron)
    cout << "\nTesting complex expression (neuron):" << endl;

    shared_ptr<Value> x1 = make_shared<Value>(2.0, set<shared_ptr<Value>>{}, "", "x1");
    shared_ptr<Value> x2 = make_shared<Value>(0.0, set<shared_ptr<Value>>{}, "", "x2");
    shared_ptr<Value> w1 = make_shared<Value>(-3.0, set<shared_ptr<Value>>{}, "", "w1");
    shared_ptr<Value> w2 = make_shared<Value>(1.0, set<shared_ptr<Value>>{}, "", "w2");
    shared_ptr<Value> b_bias = make_shared<Value>(6.8813735870195432, set<shared_ptr<Value>>{}, "", "b_bias");

    // x1*w1 + x2*w2 + b_bias
    shared_ptr<Value> x1w1 = (*x1) * w1; x1w1->label = "x1*w1";
    shared_ptr<Value> x2w2 = (*x2) * w2; x2w2->label = "x2*w2";
    shared_ptr<Value> x1w1x2w2 = (*x1w1) + x2w2; x1w1x2w2->label = "x1w1 + x2w2";
    shared_ptr<Value> n = (*x1w1x2w2) + b_bias; n->label = "n";
    shared_ptr<Value> o = n->tanh(); o->label = "o"; // Use the tanh member function

    cout << "Forward pass (neuron) complete." << endl;
    cout << "o: " << *o << endl;

    cout << "\nBackward pass (neuron)..." << endl;
    o->backward();
    cout << "Backward pass (neuron) complete." << endl;

    // Print gradients
    cout << "x1: " << *x1 << endl;
    cout << "x2: " << *x2 << endl;
    cout << "w1: " << *w1 << endl;
    cout << "w2: " << *w2 << endl;
    cout << "b_bias: " << *b_bias << endl;
    cout << "n: " << *n << endl;
    cout << "o: " << *o << endl;

    return 0;
}