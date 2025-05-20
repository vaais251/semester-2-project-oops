#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <random>

using namespace std;

class Value {
public:
    double data;
    double grad;
    function<void()> _backward;
    vector<Value*> _prev;
    string _op;
    string label;
    bool _is_heap_allocated;

    // Constructor
    Value(double data_val, const vector<Value*>& children = {}, 
          const string& op = "", const string& label = "") 
        : data(data_val), grad(0.0), _op(op), label(label), _is_heap_allocated(false)
    {
        // Add all children to _prev
        for (Value* child : children) {
            _prev.push_back(child);
        }
        
        // Default backward function does nothing
        _backward = [](){};
    }
    
    // Destructor
    ~Value() {
        // Nothing special needed here
    }
    
    // For printing the value
    void print() const {
        cout << "Value(data=" << data << ")" << endl;
    }
    
    // Helper function to create a new Value on the heap
    static Value* create(double data_val, const vector<Value*>& children = {}, 
                         const string& op = "", const string& label = "") {
        Value* v = new Value(data_val, children, op, label);
        v->_is_heap_allocated = true;
        return v;
    }
    
    // Addition operator
    Value* operator+(Value& other) {
        Value* out = Value::create(this->data + other.data, {this, &other}, "+");
        
        // Define the backward function (now using pointers)
        out->_backward = [this, &other, out]() {
            this->grad += 1.0 * out->grad;
            other.grad += 1.0 * out->grad;
        };
        
        return out;
    }
    
    // Multiplication operator
    Value* operator*(Value& other) {
        Value* out = Value::create(this->data * other.data, {this, &other}, "*");
        
        // Define the backward function
        out->_backward = [this, &other, out]() {
            this->grad += other.data * out->grad;
            other.grad += this->data * out->grad;
        };
        
        return out;
    }
    
    // Power operator
    Value* pow(double exponent) {
        Value* out = Value::create(std::pow(this->data, exponent), {this}, "**" + to_string(exponent));
        
        // Define the backward function
        out->_backward = [this, exponent, out]() {
            this->grad += exponent * std::pow(this->data, exponent - 1) * out->grad;
        };
        
        return out;
    }
    
    // Division operator (self / other)
    Value* operator/(Value& other) {
        // Division is self * other^(-1)
        return *this * (*other.pow(-1.0));
    }
    
    // Negation operator (-self)
    Value* operator-() {
        // Create a constant Value for -1
        Value neg_one(-1.0);
        return neg_one * (*this);
    }
    
    // Subtraction operator (self - other)
    Value* operator-(Value& other) {
        // Subtraction is self + (-other)
        return *this + (*(-other));
    }
    
    // Hyperbolic tangent (tanh) activation function
    Value* tanh() {
        double x = this->data;
        double t = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);
        Value* out = Value::create(t, {this}, "tanh");
        
        // Define the backward function
        out->_backward = [this, t, out]() {
            this->grad += (1 - t*t) * out->grad;
        };
        
        return out;
    }
    
    // Exponential function (e^x)
    Value* exp() {
        double x = this->data;
        double result = std::exp(x);
        Value* out = Value::create(result, {this}, "exp");
        
        // Define the backward function
        out->_backward = [this, out]() {
            this->grad += out->data * out->grad;
        };
        
        return out;
    }
    
    // Backward method to perform backpropagation
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
    
    // Helper function to clean up the entire computation graph
    static void cleanup_graph(Value* root) {
        if (!root) return;
        
        // Collect all nodes in the graph
        vector<Value*> nodes;
        unordered_set<Value*> visited;
        
        function<void(Value*)> collect_nodes = [&](Value* v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (Value* child : v->_prev) {
                    collect_nodes(child);
                }
                if (v->_is_heap_allocated) {
                    nodes.push_back(v);
                }
            }
        };
        
        collect_nodes(root);
        
        // Delete all heap-allocated nodes
        for (Value* node : nodes) {
            delete node;
        }
    }
};

class Neuron {
private:
    vector<Value*> w; // Store pointers to Value objects
    Value* b;

public:
    // Constructor - initialize with 'nin' input neurons
    Neuron(int nin) {
        // Initialize weights and bias with random values between -1 and 1
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        
        for (int i = 0; i < nin; i++) {
            w.push_back(new Value(dis(gen)));
        }
        b = new Value(dis(gen));
    }
    
    // Destructor to properly clean up memory
    ~Neuron() {
        // Cleanup all weights and bias
        for (auto& weight : w) {
            delete weight;
        }
        delete b;
    }

    // Forward pass - equivalent to __call__ in Python
    Value* operator()(vector<Value*>& x) {
        // w * x + b
        Value* act = b; // Start with bias
        
        for (size_t i = 0; i < w.size() && i < x.size(); i++) {
            // Use w[i] and x[i] which are both pointers
            Value* product = *w[i] * *x[i];
            Value* sum = *act + *product;
            act = sum; // Store the pointer
        }
        
        // Apply tanh activation function
        Value* out = act->tanh();
        return out;
    }

    // Return all parameters for optimization
    vector<Value*> parameters() {
        vector<Value*> params;
        for (auto& weight : w) {
            params.push_back(weight); // Already a pointer
        }
        params.push_back(b); // Already a pointer
        return params;
    }
};

class Layer {
private:
    vector<Neuron*> neurons;

public:
    // Constructor - initialize with nin inputs and nout outputs
    Layer(int nin, int nout) {
        for (int i = 0; i < nout; i++) {
            neurons.push_back(new Neuron(nin));
        }
    }
    
    // Destructor to properly clean up memory
    ~Layer() {
        for (auto& neuron : neurons) {
            delete neuron;
        }
    }

    // Forward pass - equivalent to __call__ in Python
    vector<Value*> operator()(vector<Value*>& x) {
        vector<Value*> outs;
        for (auto& neuron : neurons) {
            outs.push_back((*neuron)(x));
        }
        
        // If there's only one output, return it directly
        // But since C++ can't do this kind of type switching like Python,
        // we'll just return the vector and let the user handle it
        return outs;
    }

    // Return all parameters for optimization
    vector<Value*> parameters() {
        vector<Value*> params;
        for (auto& neuron : neurons) {
            vector<Value*> neuron_params = neuron->parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }
};

class MLP {
private:
    vector<Layer*> layers;

public:
    // Constructor - initialize with input size and a vector of output sizes for each layer
    MLP(int nin, const vector<int>& nouts) {
        vector<int> sizes = {nin};
        sizes.insert(sizes.end(), nouts.begin(), nouts.end());
        
        for (size_t i = 0; i < nouts.size(); i++) {
            layers.push_back(new Layer(sizes[i], sizes[i+1]));
        }
    }
    
    // Destructor to properly clean up memory
    ~MLP() {
        for (auto& layer : layers) {
            delete layer;
        }
    }
    
    // Forward pass - equivalent to __call__ in Python
    vector<Value*> operator()(vector<Value*>& x) {
        vector<Value*> activations = x;
        for (auto& layer : layers) {
            activations = (*layer)(activations);
        }
        return activations;
    }
    
    // Return all parameters for optimization
    vector<Value*> parameters() {
        vector<Value*> params;
        for (auto& layer : layers) {
            vector<Value*> layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

int main() {
    
    
    // Test case for Neuron class
    cout << "\n=== Neuron test ===\n" << endl;
    
    // Create a neuron with 3 inputs
    Neuron neuron(3);
    
    // Create input values as pointers now
    vector<Value*> x = {
        new Value(0.5), 
        new Value(-0.5), 
        new Value(1.0)
    };
    
    // Forward pass through the neuron
    Value* output = neuron(x);
    
    cout << "Neuron output: " << output->data << endl;
    
    // Backpropagation through the neuron
    output->backward();
    
    // Get parameters for inspection
    vector<Value*> params = neuron.parameters();
    
    cout << "Parameter gradients:" << endl;
    for (size_t i = 0; i < params.size(); i++) {
        if (i < params.size() - 1) {
            cout << "  Weight " << i << ": data=" << params[i]->data 
                 << ", grad=" << params[i]->grad << endl;
        } else {
            cout << "  Bias: data=" << params[i]->data 
                 << ", grad=" << params[i]->grad << endl;
        }
    }
    
    // Clean up the computation graph
    Value::cleanup_graph(output);
    
    // Clean up input values
    for (auto* val : x) {
        delete val;
    }
    
    // Test case for Layer class
    cout << "\n=== Layer test ===\n" << endl;
    
    // Create a layer with 3 inputs and 2 outputs
    Layer layer(3, 2);
    
    // Create input values
    vector<Value*> x_layer = {
        new Value(0.8), 
        new Value(-0.2), 
        new Value(0.5)
    };
    
    // Forward pass through the layer
    vector<Value*> outputs = layer(x_layer);
    
    cout << "Layer outputs:" << endl;
    for (size_t i = 0; i < outputs.size(); i++) {
        cout << "  Neuron " << i << ": " << outputs[i]->data << endl;
    }
    
    // Backpropagation through one of the outputs
    outputs[0]->backward();
    
    // Get parameters for inspection
    vector<Value*> layer_params = layer.parameters();
    
    cout << "Layer has " << layer_params.size() << " parameters" << endl;
    cout << "First few parameter gradients:" << endl;
    for (size_t i = 0; i < min(size_t(5), layer_params.size()); i++) {
        cout << "  Param " << i << ": data=" << layer_params[i]->data 
             << ", grad=" << layer_params[i]->grad << endl;
    }
    
    // Clean up the computation graph
    for (auto* output : outputs) {
        Value::cleanup_graph(output);
    }
    
    // Clean up input values
    for (auto* val : x_layer) {
        delete val;
    }
    
    // Test case for MLP class
    cout << "\n=== MLP test ===\n" << endl;
    
    // Create an MLP with 3 inputs, a hidden layer of 4 neurons, and 2 outputs
    MLP mlp(3, {4, 2});
    
    // Create input values
    vector<Value*> x_mlp = {
        new Value(0.2), 
        new Value(0.1), 
        new Value(-0.3)
    };
    
    // Forward pass through the MLP
    vector<Value*> mlp_outputs = mlp(x_mlp);
    
    cout << "MLP outputs:" << endl;
    for (size_t i = 0; i < mlp_outputs.size(); i++) {
        cout << "  Output " << i << ": " << mlp_outputs[i]->data << endl;
    }
    
    // Backpropagation through the MLP (setting gradient for both outputs)
    for (auto* output : mlp_outputs) {
        output->grad = 1.0;  // Setting gradient for all outputs
    }
    
    // Only backward on the first output to see how gradients flow
    mlp_outputs[0]->backward();
    
    // Get parameters for inspection
    vector<Value*> mlp_params = mlp.parameters();
    
    cout << "MLP has " << mlp_params.size() << " parameters" << endl;
    cout << "First few parameter gradients:" << endl;
    for (size_t i = 0; i < min(size_t(5), mlp_params.size()); i++) {
        cout << "  Param " << i << ": data=" << mlp_params[i]->data 
             << ", grad=" << mlp_params[i]->grad << endl;
    }
    
    // Clean up the computation graph
    for (auto* output : mlp_outputs) {
        Value::cleanup_graph(output);
    }
    
    // Clean up input values
    for (auto* val : x_mlp) {
        delete val;
    }
    
    return 0;
}