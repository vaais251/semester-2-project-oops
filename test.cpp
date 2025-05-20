#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>

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
        Value* out = Value::create(::pow(this->data, exponent), {this}, "**" + to_string(exponent));
        
        // Define the backward function
        out->_backward = [this, exponent, out]() {
            this->grad += exponent * ::pow(this->data, exponent - 1) * out->grad;
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
        double t = (::exp(2*x) - 1) / (::exp(2*x) + 1);
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
        double result = ::exp(x);
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
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1.0, 1.0);
        
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



// Function to load housing data from a CSV file
bool loadHousingDataCSV(const string& filename, vector<vector<double>>& X_data, vector<double>& y_data) {
    ifstream csvFile(filename);
    if (!csvFile.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return false;
    }
    
    // Clear any existing data
    X_data.clear();
    y_data.clear();
    
    string line;
    
    // Skip header line
    getline(csvFile, line);
    
    // Read data lines
    while (getline(csvFile, line)) {
        stringstream ss(line);
        string value;
        vector<double> features;
        
        // Parse the first 4 values as features
        for (int i = 0; i < 4; i++) {
            if (getline(ss, value, ',')) {
                features.push_back(stod(value));
            } else {
                cout << "Error parsing CSV line: " << line << endl;
                return false;
            }
        }
        
        // Parse the last value as the price (target)
        if (getline(ss, value, ',')) {
            y_data.push_back(stod(value));
            X_data.push_back(features);
        } else {
            cout << "Error parsing CSV line: " << line << endl;
            return false;
        }
    }
    
    csvFile.close();
    cout << "Loaded " << X_data.size() << " samples from " << filename << endl;
    return true;
}



// Function to normalize data
void normalizeData(vector<vector<double>>& X_data, vector<double>& y_data, 
                   vector<double>& X_means, vector<double>& X_stds, 
                   double& y_mean, double& y_std) {
    const int NUM_FEATURES = X_data[0].size();
    const int num_samples = X_data.size();
    
    // Initialize means and stds vectors
    X_means.resize(NUM_FEATURES, 0.0);
    X_stds.resize(NUM_FEATURES, 0.0);
    y_mean = 0.0;
    y_std = 0.0;
    
    // Calculate means
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_means[j] += X_data[i][j];
        }
        y_mean += y_data[i];
    }
    
    for (int j = 0; j < NUM_FEATURES; j++) {
        X_means[j] /= num_samples;
    }
    y_mean /= num_samples;
    
    // Calculate standard deviations
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_stds[j] += pow(X_data[i][j] - X_means[j], 2);
        }
        y_std += pow(y_data[i] - y_mean, 2);
    }
    
    for (int j = 0; j < NUM_FEATURES; j++) {
        X_stds[j] = sqrt(X_stds[j] / num_samples);
        // Prevent division by zero
        if (X_stds[j] < 1e-5) X_stds[j] = 1.0;
    }
    y_std = sqrt(y_std / num_samples);
    if (y_std < 1e-5) y_std = 1.0;
    
    // Apply normalization
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_data[i][j] = (X_data[i][j] - X_means[j]) / X_stds[j];
        }
        y_data[i] = (y_data[i] - y_mean) / y_std;
    }
    
    cout << "Data normalization complete" << endl;
}

// Function to convert standard vectors to Value* vectors for training
void prepareTrainingData(const vector<vector<double>>& X_data, const vector<double>& y_data,
                          vector<vector<Value*>>& X_train_vals, vector<Value*>& y_train_vals,
                          vector<vector<Value*>>& X_val_vals, vector<Value*>& y_val_vals) {
    const int num_samples = X_data.size();
    const int NUM_FEATURES = X_data[0].size();
    
    // Split data into training (80%) and validation (20%)
    const int TRAIN_SIZE = num_samples * 0.8;
    const int VAL_SIZE = num_samples - TRAIN_SIZE;
    
    // Initialize the Value* vectors with the right sizes
    X_train_vals.resize(TRAIN_SIZE, vector<Value*>(NUM_FEATURES));
    y_train_vals.resize(TRAIN_SIZE);
    X_val_vals.resize(VAL_SIZE, vector<Value*>(NUM_FEATURES));
    y_val_vals.resize(VAL_SIZE);
    
    // Create training data Value objects
    for (int i = 0; i < TRAIN_SIZE; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_train_vals[i][j] = new Value(X_data[i][j]);
        }
        y_train_vals[i] = new Value(y_data[i]);
    }
    
    // Create validation data Value objects
    for (int i = 0; i < VAL_SIZE; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_val_vals[i][j] = new Value(X_data[TRAIN_SIZE + i][j]);
        }
        y_val_vals[i] = new Value(y_data[TRAIN_SIZE + i]);
    }
    
    cout << "Prepared " << TRAIN_SIZE << " training samples and " 
         << VAL_SIZE << " validation samples" << endl;
}

// Function to train the model
void trainModel(MLP& mlp, 
               vector<vector<Value*>>& X_train_vals, vector<Value*>& y_train_vals,
               vector<vector<Value*>>& X_val_vals, vector<Value*>& y_val_vals,
               double& y_mean, double& y_std) {
    
    const int TRAIN_SIZE = X_train_vals.size();
    const int VAL_SIZE = X_val_vals.size();
    
    // Define training parameters
    const int EPOCHS = 50;
    const double LEARNING_RATE = 0.01;
    
    cout << "\nTraining model with " << EPOCHS << " epochs" << endl;
    cout << "Epoch, Train Loss, Validation Loss" << endl;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Reset gradients for all parameters
        vector<Value*> params = mlp.parameters();
        for (auto* p : params) {
            p->grad = 0.0;
        }
        
        // Calculate training loss (MSE)
        double train_loss = 0.0;
        for (int i = 0; i < TRAIN_SIZE; i++) {
            // Forward pass - make sure we capture the output
            vector<Value*> pred = mlp(X_train_vals[i]);
            
            // Calculate loss manually to avoid memory issues
            double error = pred[0]->data - y_train_vals[i]->data;
            train_loss += error * error;
            
            // Set gradient for output
            pred[0]->grad = 2.0 * error;
            
            // Backward pass
            pred[0]->backward();
            
            // Clean up temporary values
            Value::cleanup_graph(pred[0]);
        }
        
        train_loss /= TRAIN_SIZE;
        
        // Calculate validation loss (only every 10 epochs)
        double val_loss = 0.0;
        if (epoch % 10 == 0 || epoch == EPOCHS - 1) {
            for (int i = 0; i < VAL_SIZE; i++) {
                vector<Value*> pred = mlp(X_val_vals[i]);
                double error = pred[0]->data - y_val_vals[i]->data;
                val_loss += error * error;
                
                // Don't need backward for validation
                // Just clean up the prediction values
                for (auto* p : pred) {
                    if (p->_is_heap_allocated) {
                        delete p;
                    }
                }
            }
            val_loss /= VAL_SIZE;
        }
        
        // Apply gradients (SGD update)
        for (auto* p : params) {
            p->data -= LEARNING_RATE * p->grad;
        }
        
        // Print progress
        if (epoch % 10 == 0 || epoch == EPOCHS - 1) {
            cout << epoch + 1 << ", " << train_loss << ", " << val_loss << endl;
        }
    }
    
    // Test on some examples to see predictions
    cout << "\nPredictions on first 5 samples:" << endl;
    cout << "Actual Price, Predicted Price" << endl;
    
    for (int i = 0; i < 5 && i < TRAIN_SIZE; i++) {
        vector<Value*> pred = mlp(X_train_vals[i]);
        double predicted_price = pred[0]->data * y_std + y_mean;
        double actual_price = y_train_vals[i]->data * y_std + y_mean;
        
        cout << "$" << actual_price << ", $" << predicted_price << endl;
        
        // Clean up
        Value::cleanup_graph(pred[0]);
    }
}

// Function to clean up Value* objects to prevent memory leaks
void cleanup(vector<vector<Value*>>& X_train_vals, vector<Value*>& y_train_vals,
             vector<vector<Value*>>& X_val_vals, vector<Value*>& y_val_vals) {
    
    // Clean up training data
    for (auto& row : X_train_vals) {
        for (auto* val : row) {
            delete val;
        }
    }
    for (auto* val : y_train_vals) {
        delete val;
    }
    
    // Clean up validation data
    for (auto& row : X_val_vals) {
        for (auto* val : row) {
            delete val;
        }
    }
    for (auto* val : y_val_vals) {
        delete val;
    }
    
    X_train_vals.clear();
    y_train_vals.clear();
    X_val_vals.clear();
    y_val_vals.clear();
}

int main() {
    cout << "\n=== Housing Price Prediction with MLP ===\n" << endl;
    
    // Data containers
    const int NUM_FEATURES = 4;
    vector<vector<double>> X_data;
    vector<double> y_data;
    string csv_filename = "housing_data.csv";
    
    // Load data from the CSV file
    cout << "Loading housing data from CSV..." << endl;
    if (!loadHousingDataCSV(csv_filename, X_data, y_data)) {
        cout << "Failed to load data from CSV file: " << csv_filename << endl;
        cout << "Please make sure the file exists in the current directory." << endl;
        return 1;
    }
    
    // Print sample data
    const int num_samples = X_data.size();
    cout << "\nLoaded " << num_samples << " housing records" << endl;
    cout << "\nSample data (first 5 records):" << endl;
    cout << "SQFT, Bedrooms, Bathrooms, Age (years), Price ($)" << endl;
    for (int i = 0; i < 5 && i < num_samples; i++) {
        cout << X_data[i][0] << ", " << X_data[i][1] << ", " << X_data[i][2] << ", " 
             << X_data[i][3] << ", $" << y_data[i] << endl;
    }
    
    // Normalize the data
    vector<double> X_means, X_stds;
    double y_mean, y_std;
    cout << "\nNormalizing data..." << endl;
    normalizeData(X_data, y_data, X_means, X_stds, y_mean, y_std);
    
    // Prepare training data
    vector<vector<Value*>> X_train_vals, X_val_vals;
    vector<Value*> y_train_vals, y_val_vals;
    cout << "\nPreparing training and validation sets..." << endl;
    prepareTrainingData(X_data, y_data, X_train_vals, y_train_vals, X_val_vals, y_val_vals);
    
    // Create neural network
    cout << "\nCreating neural network..." << endl;
    MLP mlp(NUM_FEATURES, {8, 4, 1});
    
    // Train model
    trainModel(mlp, X_train_vals, y_train_vals, X_val_vals, y_val_vals, y_mean, y_std);
    
    // Clean up allocated memory
    cout << "\nCleaning up resources..." << endl;
    cleanup(X_train_vals, y_train_vals, X_val_vals, y_val_vals);
    
    cout << "\nTraining complete." << endl;
    return 0;
}