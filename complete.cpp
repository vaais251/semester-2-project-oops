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
    static vector<Value*> computation_graph; // Track all created values

    Value(double data_val, const vector<Value*>& children = {}, 
          const string& op = "", const string& label = "") 
        : data(data_val), grad(0.0), _op(op), label(label), _is_heap_allocated(false)
    {
        for (Value* child : children) {
            _prev.push_back(child);
        }
        _backward = [](){};
    }
    
    ~Value() {}
    
    void print() const {
        cout << "Value(data=" << data << ", grad=" << grad << ")" << endl;
    }
    
    static Value* create(double data_val, const vector<Value*>& children = {}, 
                         const string& op = "", const string& label = "") {
        Value* v = new Value(data_val, children, op, label);
        v->_is_heap_allocated = true;
        computation_graph.push_back(v);
        return v;
    }
    
    Value* operator+(Value& other) {
        Value* out = Value::create(this->data + other.data, {this, &other}, "+");
        
        out->_backward = [this, &other, out]() {
            this->grad += 1.0 * out->grad;
            other.grad += 1.0 * out->grad;
        };
        
        return out;
    }
    
    Value* operator*(Value& other) {
        Value* out = Value::create(this->data * other.data, {this, &other}, "*");
        
        out->_backward = [this, &other, out]() {
            this->grad += other.data * out->grad;
            other.grad += this->data * out->grad;
        };
        
        return out;
    }
    
    Value* pow(double exponent) {
        Value* out = Value::create(::pow(this->data, exponent), {this}, "**" + to_string(exponent));
        
        out->_backward = [this, exponent, out]() {
            this->grad += exponent * ::pow(this->data, exponent - 1) * out->grad;
        };
        
        return out;
    }
    
    Value* operator/(Value& other) {
        return *this * (*other.pow(-1.0));
    }
    
    Value* operator-() {
        Value* neg_one = Value::create(-1.0);
        return *neg_one * *this;
    }
    
    Value* operator-(Value& other) {
        return *this + (*(-other));
    }
    
    Value* tanh() {
        double x = this->data;
        double t = (::exp(2*x) - 1) / (::exp(2*x) + 1);
        Value* out = Value::create(t, {this}, "tanh");
        
        out->_backward = [this, t, out]() {
            this->grad += (1 - t*t) * out->grad;
        };
        
        return out;
    }
    
    Value* relu() {
        double result = max(0.0, this->data);
        Value* out = Value::create(result, {this}, "relu");
        
        out->_backward = [this, out]() {
            this->grad += (this->data > 0 ? 1.0 : 0.0) * out->grad;
        };
        
        return out;
    }
    
    Value* exp() {
        double x = this->data;
        double result = ::exp(x);
        Value* out = Value::create(result, {this}, "exp");
        
        out->_backward = [this, out]() {
            this->grad += out->data * out->grad;
        };
        
        return out;
    }
    
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
    
    static void clear_computation_graph() {
        for (Value* v : computation_graph) {
            delete v;
        }
        computation_graph.clear();
    }
    
    static void zero_gradients() {
        for (Value* v : computation_graph) {
            v->grad = 0.0;
        }
    }
};

vector<Value*> Value::computation_graph;

class Neuron {
private:
    vector<Value*> w;
    Value* b;

public:
    Neuron(int nin) {
        random_device rd;
        mt19937 gen(rd());
        // Improved Xavier initialization
        uniform_real_distribution<> dis(-1.0, 1.0);
        double xavier_scale = sqrt(2.0 / nin); // Better for ReLU
        
        for (int i = 0; i < nin; i++) {
            w.push_back(new Value(dis(gen) * xavier_scale));
        }
        b = new Value(0.0); // Start with zero bias
    }
    
    ~Neuron() {
        for (auto& weight : w) {
            delete weight;
        }
        delete b;
    }

    Value* operator()(vector<Value*>& x) {
        Value* act = b;
        
        for (size_t i = 0; i < w.size() && i < x.size(); i++) {
            Value* product = *w[i] * *x[i];
            Value* sum = *act + *product;
            act = sum;
        }
        
        // Use ReLU for hidden layers (will be handled by Layer class)
        return act;
    }

    vector<Value*> parameters() {
        vector<Value*> params;
        for (auto& weight : w) {
            params.push_back(weight);
        }
        params.push_back(b);
        return params;
    }
};

class Layer {
private:
    vector<Neuron*> neurons;
    bool is_output_layer;

public:
    Layer(int nin, int nout, bool output_layer = false) : is_output_layer(output_layer) {
        for (int i = 0; i < nout; i++) {
            neurons.push_back(new Neuron(nin));
        }
    }
    
    ~Layer() {
        for (auto& neuron : neurons) {
            delete neuron;
        }
    }

    vector<Value*> operator()(vector<Value*>& x) {
        vector<Value*> outs;
        for (auto& neuron : neurons) {
            Value* raw_output = (*neuron)(x);
            // Apply activation function based on layer type
            if (is_output_layer) {
                // No activation for output layer (linear output for regression)
                outs.push_back(raw_output);
            } else {
                // ReLU for hidden layers
                outs.push_back(raw_output->relu());
            }
        }
        return outs;
    }

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
    MLP(int nin, const vector<int>& nouts) {
        vector<int> sizes = {nin};
        sizes.insert(sizes.end(), nouts.begin(), nouts.end());
        
        for (size_t i = 0; i < nouts.size(); i++) {
            bool is_last_layer = (i == nouts.size() - 1);
            layers.push_back(new Layer(sizes[i], sizes[i+1], is_last_layer));
        }
        
        cout << "Created MLP with architecture: " << nin;
        for (auto size : nouts) {
            cout << "-" << size;
        }
        cout << endl;
    }
    
    ~MLP() {
        for (auto& layer : layers) {
            delete layer;
        }
    }
    
    vector<Value*> operator()(vector<Value*>& x) {
        vector<Value*> activations = x;
        for (auto& layer : layers) {
            activations = (*layer)(activations);
        }
        return activations;
    }
    
    vector<Value*> parameters() {
        vector<Value*> params;
        for (auto& layer : layers) {
            vector<Value*> layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

bool loadHousingDataCSV(const string& filename, vector<vector<double>>& X_data, vector<double>& y_data) {
    ifstream csvFile(filename);
    if (!csvFile.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return false;
    }
    
    X_data.clear();
    y_data.clear();
    
    string line;
    getline(csvFile, line); // Skip header
    
    while (getline(csvFile, line)) {
        stringstream ss(line);
        string value;
        vector<double> features;
        
        for (int i = 0; i < 4; i++) {
            if (getline(ss, value, ',')) {
                features.push_back(stod(value));
            } else {
                cout << "Error parsing CSV line: " << line << endl;
                return false;
            }
        }
        
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

void normalizeData(vector<vector<double>>& X_data, vector<double>& y_data, 
                   vector<double>& X_means, vector<double>& X_stds, 
                   double& y_mean, double& y_std) {
    const int NUM_FEATURES = X_data[0].size();
    const int num_samples = X_data.size();
    
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
        if (X_stds[j] < 1e-8) X_stds[j] = 1.0; // Prevent division by zero
    }
    y_std = sqrt(y_std / num_samples);
    if (y_std < 1e-8) y_std = 1.0;
    
    // Normalize the data
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_data[i][j] = (X_data[i][j] - X_means[j]) / X_stds[j];
        }
        y_data[i] = (y_data[i] - y_mean) / y_std;
    }
    
    cout << "Data normalization complete" << endl;
    cout << "Target mean: " << y_mean << ", Target std: " << y_std << endl;
}

void prepareTrainingData(const vector<vector<double>>& X_data, const vector<double>& y_data,
                          vector<vector<Value*>>& X_train_vals, vector<Value*>& y_train_vals,
                          vector<vector<Value*>>& X_val_vals, vector<Value*>& y_val_vals) {
    const int num_samples = X_data.size();
    const int NUM_FEATURES = X_data[0].size();
    
    const int TRAIN_SIZE = num_samples * 0.8;
    const int VAL_SIZE = num_samples - TRAIN_SIZE;
    
    X_train_vals.resize(TRAIN_SIZE, vector<Value*>(NUM_FEATURES));
    y_train_vals.resize(TRAIN_SIZE);
    X_val_vals.resize(VAL_SIZE, vector<Value*>(NUM_FEATURES));
    y_val_vals.resize(VAL_SIZE);
    
    for (int i = 0; i < TRAIN_SIZE; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_train_vals[i][j] = new Value(X_data[i][j]);
        }
        y_train_vals[i] = new Value(y_data[i]);
    }
    
    for (int i = 0; i < VAL_SIZE; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            X_val_vals[i][j] = new Value(X_data[TRAIN_SIZE + i][j]);
        }
        y_val_vals[i] = new Value(y_data[TRAIN_SIZE + i]);
    }
    
    cout << "Prepared " << TRAIN_SIZE << " training samples and " 
         << VAL_SIZE << " validation samples" << endl;
}

void trainModel(MLP& mlp, 
               vector<vector<Value*>>& X_train_vals, vector<Value*>& y_train_vals,
               vector<vector<Value*>>& X_val_vals, vector<Value*>& y_val_vals,
               double& y_mean, double& y_std) {
    
    const int TRAIN_SIZE = X_train_vals.size();
    const int VAL_SIZE = X_val_vals.size();
    
    const int EPOCHS = 200;
    const double INITIAL_LR = 0.01;
    const double WEIGHT_DECAY = 1e-6;
    
    cout << "\nTraining model with " << EPOCHS << " epochs" << endl;
    cout << "Epoch, Train Loss, Validation Loss, Learning Rate" << endl;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Learning rate decay
        double learning_rate = INITIAL_LR * pow(0.995, epoch);
        
        // Clear gradients for all parameters
        vector<Value*> params = mlp.parameters();
        for (auto* p : params) {
            p->grad = 0.0;
        }
        
        // Clear computation graph from previous epoch
        Value::clear_computation_graph();
        
        double total_loss = 0.0;
        
        // Training loop
        for (int i = 0; i < TRAIN_SIZE; i++) {
            vector<Value*> pred = mlp(X_train_vals[i]);
            
            // Calculate MSE loss
            Value* diff = *pred[0] - *y_train_vals[i];
            Value* loss = diff->pow(2.0);
            total_loss += loss->data;
            
            // Backward pass
            loss->backward();
        }
        
        double train_loss = total_loss / TRAIN_SIZE;
        
        // Validation loss calculation
        double val_loss = 0.0;
        if (epoch % 20 == 0 || epoch == EPOCHS - 1) {
            Value::clear_computation_graph();
            
            for (int i = 0; i < VAL_SIZE; i++) {
                vector<Value*> pred = mlp(X_val_vals[i]);
                Value* diff = *pred[0] - *y_val_vals[i];
                Value* loss = diff->pow(2.0);
                val_loss += loss->data;
            }
            val_loss /= VAL_SIZE;
            
            Value::clear_computation_graph();
        }
        
        // Parameter update with gradient clipping
        double grad_norm = 0.0;
        for (auto* p : params) {
            grad_norm += p->grad * p->grad;
        }
        grad_norm = sqrt(grad_norm);
        
        double clip_value = 1.0;
        if (grad_norm > clip_value) {
            for (auto* p : params) {
                p->grad *= clip_value / grad_norm;
            }
        }
        
        for (auto* p : params) {
            p->data -= learning_rate * (p->grad + WEIGHT_DECAY * p->data);
        }
        
        if (epoch % 20 == 0 || epoch == EPOCHS - 1) {
            cout << epoch + 1 << ", " << train_loss << ", " << val_loss 
                 << ", " << learning_rate << endl;
        }
    }
    
    // Final predictions
    cout << "\nPredictions on first 5 training samples:" << endl;
    cout << "Actual Price, Predicted Price, Error" << endl;
    
    Value::clear_computation_graph();
    
    for (int i = 0; i < 5 && i < TRAIN_SIZE; i++) {
        vector<Value*> pred = mlp(X_train_vals[i]);
        double predicted_price = pred[0]->data * y_std + y_mean;
        double actual_price = y_train_vals[i]->data * y_std + y_mean;
        double error = abs(predicted_price - actual_price);
        
        cout << "$" << (int)actual_price << ", $" << (int)predicted_price 
             << ", $" << (int)error << endl;
    }
    
    Value::clear_computation_graph();
}

void cleanup(vector<vector<Value*>>& X_train_vals, vector<Value*>& y_train_vals,
             vector<vector<Value*>>& X_val_vals, vector<Value*>& y_val_vals) {
    
    for (auto& row : X_train_vals) {
        for (auto* val : row) {
            delete val;
        }
    }
    for (auto* val : y_train_vals) {
        delete val;
    }
    
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
    
    const int NUM_FEATURES = 4;
    vector<vector<double>> X_data;
    vector<double> y_data;
    string csv_filename = "housing_data.csv";
    
    cout << "Loading housing data from CSV..." << endl;
    if (!loadHousingDataCSV(csv_filename, X_data, y_data)) {
        cout << "Failed to load data from CSV file: " << csv_filename << endl;
        cout << "Please make sure the file exists in the current directory." << endl;
        return 1;
    }
    
    const int num_samples = X_data.size();
    cout << "\nLoaded " << num_samples << " housing records" << endl;
    cout << "\nSample data (first 5 records):" << endl;
    cout << "SQFT, Bedrooms, Bathrooms, Age (years), Price ($)" << endl;
    for (int i = 0; i < 5 && i < num_samples; i++) {
        cout << X_data[i][0] << ", " << X_data[i][1] << ", " << X_data[i][2] << ", " 
             << X_data[i][3] << ", $" << y_data[i] << endl;
    }
    
    vector<double> X_means, X_stds;
    double y_mean, y_std;
    cout << "\nNormalizing data..." << endl;
    normalizeData(X_data, y_data, X_means, X_stds, y_mean, y_std);
    
    vector<vector<Value*>> X_train_vals, X_val_vals;
    vector<Value*> y_train_vals, y_val_vals;
    cout << "\nPreparing training and validation sets..." << endl;
    prepareTrainingData(X_data, y_data, X_train_vals, y_train_vals, X_val_vals, y_val_vals);
    
    cout << "\nCreating neural network..." << endl;
    // Better architecture: 4 -> 16 -> 8 -> 1
    MLP mlp(NUM_FEATURES, {16, 8, 1});
    
    trainModel(mlp, X_train_vals, y_train_vals, X_val_vals, y_val_vals, y_mean, y_std);
    
    cout << "\nCleaning up resources..." << endl;
    cleanup(X_train_vals, y_train_vals, X_val_vals, y_val_vals);
    Value::clear_computation_graph();
    
    cout << "\nTraining complete." << endl;
    return 0;
}