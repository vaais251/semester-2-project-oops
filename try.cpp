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
        for (Value* child : children) {
            _prev.push_back(child);
        }
        _backward = [](){};
    }
    
    // Destructor
    ~Value() {
        // Nothing special needed here for now
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
        out->_backward = [this, &other, out]() {
            this->grad += 1.0 * out->grad;
            other.grad += 1.0 * out->grad;
        };
        return out;
    }
    
    // Multiplication operator
    Value* operator*(Value& other) {
        Value* out = Value::create(this->data * other.data, {this, &other}, "*");
        out->_backward = [this, &other, out]() {
            this->grad += other.data * out->grad;
            other.grad += this->data * out->grad;
        };
        return out;
    }
    
    // Power operator
    Value* pow(double exponent) {
        Value* out = Value::create(std::pow(this->data, exponent), {this}, "**" + to_string(exponent));
        out->_backward = [this, exponent, out]() {
            this->grad += exponent * std::pow(this->data, exponent - 1) * out->grad;
        };
        return out;
    }
    
    // Division operator (self / other)
    Value* operator/(Value& other) {
        // Division is self * other^(-1)
        return *this * *(other.pow(-1.0));
    }
    
    // Negation operator (-self)
    Value* operator-() {
        Value neg_one(-1.0);
        return neg_one * (*this);
    }
    
    // Subtraction operator (self - other)
    Value* operator-(Value& other) {
        return *this + (*(-other));
    }
    
    // Hyperbolic tangent activation
    Value* tanh() {
        double x = this->data;
        double t = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);
        Value* out = Value::create(t, {this}, "tanh");
        out->_backward = [this, t, out]() {
            this->grad += (1 - t*t) * out->grad;
        };
        return out;
    }
    
    // Exponential function
    Value* exp() {
        double x = this->data;
        double result = std::exp(x);
        Value* out = Value::create(result, {this}, "exp");
        out->_backward = [this, out]() {
            this->grad += out->data * out->grad;
        };
        return out;
    }
    
    // Backward pass to compute gradients
    void backward() {
        vector<Value*> topo;
        unordered_set<Value*> visited;
        function<void(Value*)> build_topo = [&](Value* v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (Value* child : v->_prev) {
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };
        build_topo(this);
        this->grad = 1.0;
        reverse(topo.begin(), topo.end());
        for (Value* node : topo) {
            node->_backward();
        }
    }
    
    // Cleanup helper to delete all heap allocated nodes reachable from root
    static void cleanup_graph(Value* root) {
        if (!root) return;
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
        for (Value* node : nodes) {
            delete node;
        }
    }
};

class Neuron {
public:
    vector<Value*> w;
    Value* b;

    Neuron(int nin) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1.0, 1.0);
        for (int i = 0; i < nin; ++i) {
            w.push_back(Value::create(dis(gen)));
        }
        b = Value::create(dis(gen));
    }

    Value* operator()(const vector<Value*>& x) {
        Value* act = b;
        for (size_t i = 0; i < w.size(); ++i) {
            Value* prod = *w[i] * *x[i];
            Value* new_act = *act + *prod;
            act = new_act;
        }
        return act->tanh();
    }

    vector<Value*> parameters() {
        vector<Value*> params = w;
        params.push_back(b);
        return params;
    }

    ~Neuron() {
        for (Value* weight : w) {
            if (weight->_is_heap_allocated) delete weight;
        }
        if (b->_is_heap_allocated) delete b;
    }
};

class Layer {
public:
    vector<Neuron*> neurons;

    Layer(int nin, int nout) {
        for (int i = 0; i < nout; ++i) {
            neurons.push_back(new Neuron(nin));
        }
    }

    vector<Value*> operator()(const vector<Value*>& x) {
        vector<Value*> outs;
        for (Neuron* n : neurons) {
            outs.push_back((*n)(x));
        }
        return outs;
    }

    vector<Value*> parameters() {
        vector<Value*> params;
        for (Neuron* n : neurons) {
            vector<Value*> n_params = n->parameters();
            params.insert(params.end(), n_params.begin(), n_params.end());
        }
        return params;
    }

    ~Layer() {
        for (Neuron* n : neurons) {
            delete n;
        }
    }
};

class MLP {
public:
    vector<Layer> layers;

    MLP(int nin, const vector<int>& nouts) {
        vector<int> sizes = {nin};
        sizes.insert(sizes.end(), nouts.begin(), nouts.end());
        for (size_t i = 0; i < nouts.size(); ++i) {
            layers.emplace_back(Layer(sizes[i], sizes[i+1]));
        }
    }

    vector<Value*> operator()(const vector<Value*>& x) {
        vector<Value*> out = x;
        for (Layer& layer : layers) {
            out = layer(out);
        }
        return out;
    }

    vector<Value*> parameters() {
        vector<Value*> params;
        for (Layer& layer : layers) {
            vector<Value*> layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

int main() {
    Value* in1 = Value::create(1.0);
    Value* in2 = Value::create(2.0);
    vector<Value*> input = {in1, in2};

    MLP mlp(2, {3, 1});  // 2 inputs -> hidden layer with 3 neurons -> output layer with 1 neuron
    vector<Value*> out = mlp(input);

    // Backpropagate from output
    out[0]->backward();

    cout << "Output: " << out[0]->data << endl;
    cout << "Input gradients: in1.grad=" << in1->grad << ", in2.grad=" << in2->grad << endl;

    vector<Value*> params = mlp.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        cout << "param[" << i << "] = " << params[i]->data
             << ", grad = " << params[i]->grad << endl;
    }

    // Clean up dynamically allocated Values
    for (Value* p : params) {
        if (p->_is_heap_allocated) delete p;
    }
    delete in1;
    delete in2;
    Value::cleanup_graph(out[0]);

    return 0;
}
