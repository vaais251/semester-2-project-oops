#include <iostream>
#include <vector>
#include <functional>
#include <string>

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
};

int main() {
    // Example usage
    Value a(2.0, {}, "", "a");
    Value b(3.0, {}, "", "b");
    
    Value c = a + b;
    
    // Print values
    cout << "a: ";
    a.print();
    cout << "b: ";
    b.print();
    cout << "c = a + b: ";
    c.print();
    
    return 0;
}