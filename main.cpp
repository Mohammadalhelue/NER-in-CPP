#include <bits/stdc++.h>

using namespace std;

const double LOG_ZERO = -numeric_limits<double>::infinity();
const double LAPLACE_SMOOTHING = 1e-5;
const int UNK_THRESHOLD = 1;

struct Sequence {
    vector<pair<string, string>> tokens;
};

struct HMM {
    vector<string> states;
    vector<string> vocab;
    vector<double> initial_probs;
    vector<vector<double>> trans_probs;
    vector<vector<double>> emit_probs;
    unordered_map<string, int> state_index;
    unordered_map<string, int> word_index;
};


pair<string, string> GetWordTag(string line) {
    reverse(line.begin(), line.end());
    string current = "";
    string tag, word;
    int counter = 0;
    for (int i = 0; i < line.size(); ++i) {
        if (counter % 2 == 0) {
            if (line[i] == ',') {
                if (counter == 0)
                    tag = current, current = "";
                if (counter == 2) {
                    word = current;
                    break;
                }
            } else current += line[i];
        }
        if (line[i] == ',')
            counter++;
    }
    reverse(tag.begin(), tag.end());
    reverse(word.begin(), word.end());
    return {word, tag};
}

// Load dataset from CSV file
vector<Sequence> load_dataset(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) throw runtime_error("Could not open file: " + filename);

    vector<Sequence> sequences;
    Sequence current_seq;
    string line;

    while (getline(file, line)) {
        if (line[0] == 'S') {
            if (!current_seq.tokens.empty()) {
                auto [word, tag] = GetWordTag(line);
                current_seq.tokens.push_back({word, tag});
                sequences.push_back(current_seq);
                current_seq.tokens.clear();
            }
            continue;
        }
        auto [word, tag] = GetWordTag(line);

        current_seq.tokens.push_back({word, tag});
    }

    if (!current_seq.tokens.empty()) {
        sequences.push_back(current_seq);
    }

    return sequences;
}

// Split dataset into train and test sets
void split_dataset(const vector<Sequence> &all_data, vector<Sequence> &train_set, vector<Sequence> &test_set,
                   double test_ratio = 0.2) {
    if (all_data.empty()) return;

    // Create shuffled indices
    vector<size_t> indices(all_data.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    // Calculate split point
    const size_t split_idx = static_cast<size_t>(all_data.size() * (1.0 - test_ratio));

    // Create datasets
    train_set.clear();
    test_set.clear();

    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < split_idx) {
            train_set.push_back(all_data[indices[i]]);
        } else {
            test_set.push_back(all_data[indices[i]]);
        }
    }
}


// Train HMM from dataset
HMM train_hmm(const vector<Sequence> &sequences) {
    HMM model;

    // Collect word and tag frequencies
    unordered_map<string, int> word_counts;
    unordered_map<string, int> tag_counts;

    for (const auto &seq: sequences) {
        for (const auto &token: seq.tokens) {
            word_counts[token.first]++;
            tag_counts[token.second]++;
        }
    }

    // Build vocabulary (replace rare words with <UNK>)
    model.vocab.push_back("<UNK>");
    for (const auto &w: word_counts) {
        if (w.second >= UNK_THRESHOLD) {
            model.vocab.push_back(w.first);
        }
    }

    // Build state list
    for (const auto &t: tag_counts) {
        model.states.push_back(t.first);
    }

    // Create index mappings
    for (int i = 0; i < model.states.size(); ++i)
        model.state_index[model.states[i]] = i;

    for (int i = 0; i < model.vocab.size(); ++i)
        model.word_index[model.vocab[i]] = i;

    // Initialize count matrices
    //vocad == word unique
    //states == tag unique
    const int num_states = model.states.size();
    const int vocab_size = model.vocab.size();
    vector<int> initial_counts(num_states, 0);
    vector<vector<int>> trans_counts(num_states, vector<int>(num_states, 0));
    vector<vector<int>> emit_counts(num_states, vector<int>(vocab_size, 0));


    for (int i = 0; i < sequences.size(); i++) {

        const auto &seq = sequences[i];

        for (int t = 0; t < seq.tokens.size(); ++t) {
            const string &word = seq.tokens[t].first;
            const string &tag = seq.tokens[t].second;

            // Handle unknown words
            int word_idx = (model.word_index.find(word) != model.word_index.end())
                           ? model.word_index[word]
                           : model.word_index["<UNK>"];
            int tag_idx = model.state_index[tag];

            if (t == 0) {
                initial_counts[tag_idx]++;
            } else {
                int prev_tag = model.state_index[seq.tokens[t - 1].second];
                trans_counts[prev_tag][tag_idx]++;
            }
            emit_counts[tag_idx][word_idx]++;
        }
    }

    // Convert counts to log probabilities with Laplace smoothing
    model.initial_probs.resize(num_states);
    model.trans_probs.resize(num_states, vector<double>(num_states));
    model.emit_probs.resize(num_states, vector<double>(vocab_size));

    // Initial probabilities
    double init_total = accumulate(initial_counts.begin(), initial_counts.end(), 0.0)
                        + num_states * LAPLACE_SMOOTHING;
    for (int i = 0; i < num_states; ++i) {
        double prob = (initial_counts[i] + LAPLACE_SMOOTHING) / init_total;
        model.initial_probs[i] = log(prob);
    }

    // Transition probabilities
    for (int i = 0; i < num_states; ++i) {
        double row_total = accumulate(trans_counts[i].begin(), trans_counts[i].end(), 0.0)
                           + num_states * LAPLACE_SMOOTHING;
        for (int j = 0; j < num_states; ++j) {
            double prob = (trans_counts[i][j] + LAPLACE_SMOOTHING) / row_total;
            model.trans_probs[i][j] = log(prob);
        }
    }

    // Emission probabilities
    for (int i = 0; i < num_states; ++i) {
        double row_total = accumulate(emit_counts[i].begin(), emit_counts[i].end(), 0.0)
                           + vocab_size * LAPLACE_SMOOTHING;
        for (int j = 0; j < vocab_size; ++j) {
            double prob = (emit_counts[i][j] + LAPLACE_SMOOTHING) / row_total;
            model.emit_probs[i][j] = log(prob);
        }
    }

    return model;
}

// Viterbi decoding for sequence labeling
vector<string> viterbi_decode(const HMM &model, const vector<string> &tokens) {
    const int T = tokens.size();
    const int N = model.states.size();

    // DP tables
    vector<vector<double>> viterbi(N, vector<double>(T, LOG_ZERO));
    vector<vector<int>> backpointers(N, vector<int>(T, -1));

    // Initialize first observation
    for (int s = 0; s < N; ++s) {
        auto word_iter = model.word_index.find(tokens[0]);
        int word_idx = (word_iter != model.word_index.end())
                       ? word_iter->second
                       : model.word_index.at("<UNK>");
        viterbi[s][0] = model.initial_probs[s] + model.emit_probs[s][word_idx];
    }

    // Forward pass
    for (int t = 1; t < T; ++t) {
        auto word_iter = model.word_index.find(tokens[t]);
        int word_idx = (word_iter != model.word_index.end())
                       ? word_iter->second
                       : model.word_index.at("<UNK>");

        for (int s = 0; s < N; ++s) {
            double max_logprob = LOG_ZERO;
            int best_prev = -1;

            for (int prev_s = 0; prev_s < N; ++prev_s) {
                double logprob = viterbi[prev_s][t - 1] + model.trans_probs[prev_s][s];
                if (logprob > max_logprob) {
                    max_logprob = logprob;
                    best_prev = prev_s;
                }
            }

            viterbi[s][t] = max_logprob + model.emit_probs[s][word_idx];
            backpointers[s][t] = best_prev;
        }
    }

    // Backtracking
    vector<string> best_path(T);
    int best_final = 0;
    double max_final_prob = viterbi[0][T - 1];

    for (int s = 1; s < N; ++s) {
        if (viterbi[s][T - 1] > max_final_prob) {
            max_final_prob = viterbi[s][T - 1];
            best_final = s;
        }
    }

    best_path[T - 1] = model.states[best_final];
    for (int t = T - 2; t >= 0; --t) {
        best_final = backpointers[best_final][t + 1];
        best_path[t] = model.states[best_final];
    }

    return best_path;
}

// Calculate evaluation metrics
void evaluate_model(const HMM &model, const vector<Sequence> &test_set) {
    // Evaluation metrics
    int total_tokens = 0;
    int correct_tokens = 0;
    unordered_map<string, int> true_positive;
    unordered_map<string, int> false_positive;
    unordered_map<string, int> false_negative;
    set<string> all_entities;

    // Collect all entity types
    for (const auto &seq: test_set) {
        for (const auto &token: seq.tokens) {
            const string &tag = token.second;
            if (tag != "O" && tag.find('-') != string::npos) {
                string entity = tag.substr(2);  // Remove BIO prefix
                all_entities.insert(entity);
            }
        }
    }
    // Initialize counts
    for (const auto &entity: all_entities) {
        true_positive[entity] = 0;
        false_positive[entity] = 0;
        false_negative[entity] = 0;
    }

    // Process each sequence
    for (const auto &seq: test_set) {
        // Extract words
        vector<string> words;
        for (const auto &token: seq.tokens) {
            words.push_back(token.first);
        }

        // Predict tags
        vector<string> pred_tags = viterbi_decode(model, words);

        // Compare predictions with ground truth
        for (size_t i = 0; i < seq.tokens.size(); i++) {
            total_tokens++;
            const string &true_tag = seq.tokens[i].second;
            const string &pred_tag = pred_tags[i];

            // Token-level accuracy
            if (true_tag == pred_tag) {
                correct_tokens++;
            }

            // Entity-level metrics (BIO scheme)
            if (true_tag != "O" || pred_tag != "O") {
                // Extract entity types
                string true_entity = (true_tag.find('-') != string::npos) ? true_tag.substr(2) : "";
                string pred_entity = (pred_tag.find('-') != string::npos) ? pred_tag.substr(2) : "";

                // Check if same entity
                bool same_entity = (true_entity == pred_entity) && !true_entity.empty();

                // Check if both are beginning of entity or inside
                if (same_entity) {
                    // Handle beginning tags
                    if (true_tag[0] == 'B' && pred_tag[0] == 'B') {
                        true_positive[true_entity]++;
                    }
                        // Handle inside tags - only count if previous tag matches
                    else if (true_tag[0] == 'I' && pred_tag[0] == 'I' && i > 0) {
                        const string &prev_true = seq.tokens[i - 1].second;
                        const string &prev_pred = pred_tags[i - 1];

                        // Only count if previous tag was same entity
                        if (prev_true == prev_pred) {
                            string prev_true_entity = (prev_true.find('-') != string::npos)
                                                      ? prev_true.substr(2) : "";
                            if (prev_true_entity == true_entity) {
                                true_positive[true_entity]++;
                            }
                        }
                    }
                }

                // False positives/negatives
                if (true_tag == "O" && pred_tag != "O") {
                    false_positive[pred_entity]++;
                }
                if (true_tag != "O" && pred_tag == "O") {
                    false_negative[true_entity]++;
                }
                if (!true_entity.empty() && !pred_entity.empty() && true_entity != pred_entity) {
                    false_positive[pred_entity]++;
                    false_negative[true_entity]++;
                }
            }
        }
    }

    // Calculate and print metrics
    double accuracy = (total_tokens > 0) ? static_cast<double>(correct_tokens) / total_tokens : 0.0;
    cout << "\nEvaluation Results:\n";
    cout << "=========================================\n";
    cout << "Token-level Accuracy: " << fixed << setprecision(4)
         << accuracy * 100 << "%\n";
    cout << "Total Tokens: " << total_tokens << "\n";
    cout << "Correct Tokens: " << correct_tokens << "\n\n";

    cout << "Entity-level Performance:\n";
    cout << "-----------------------------------------\n";
    cout << setw(10) << "Entity" << setw(12) << "Precision"
         << setw(10) << "Recall" << setw(10) << "F1" << "\n";
    cout << "-----------------------------------------\n";

    double macro_f1 = 0.0;
    int entity_count = 0;

    for (const auto &entity: all_entities) {
        double precision = 0.0;
        double recall = 0.0;
        double f1 = 0.0;
        int tp = true_positive[entity];
        int fp = false_positive[entity];
        int fn = false_negative[entity];

        if (tp + fp > 0) {
            precision = static_cast<double>(tp) / (tp + fp);
        }
        if (tp + fn > 0) {
            recall = static_cast<double>(tp) / (tp + fn);
        }
        if (precision > 0 || recall > 0) {
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10);
        }

        macro_f1 += f1;
        entity_count++;

        cout << setw(10) << entity
             << setw(12) << setprecision(4) << precision
             << setw(10) << recall
             << setw(10) << f1 << "\n";
    }

    if (entity_count > 0) {
        macro_f1 /= entity_count;
        cout << "\nMacro-average F1: " << setprecision(4) << macro_f1 << "\n";
    }
    cout << "=========================================\n";
}

// Interactive testing function
void test_model_interactive(const HMM &model) {
    cout << "\nEnter sentence to analyze (type 'exit' to quit):\n";
    string line;
    cin.ignore();  // Clear newline from previous input

    while (true) {
        cout << "\n> ";
        getline(cin, line);
        if (line == "exit") break;

        vector<string> tokens;
        stringstream ss(line);
        string token;
        while (ss >> token) tokens.push_back(token);

        if (tokens.empty()) continue;

        auto start = chrono::high_resolution_clock::now();
        vector<string> tags = viterbi_decode(model, tokens);
        auto end = chrono::high_resolution_clock::now();

        cout << "Entities:\n";
        for (int i = 0; i < tokens.size(); ++i) {
            cout << setw(15) << tokens[i] << " : " << tags[i] << "\n";
        }

        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "\nProcessed in " << duration.count() << "ms\n";
    }
}

int main() {
    try {
        const string data_file = "ner.csv";

        vector<Sequence> all_data;
        vector<Sequence> train_set;
        vector<Sequence> test_set;

        cout << "Loading dataset from " << data_file << "...\n";
        all_data = load_dataset(data_file);
        cout << "Inter number for splitting dataset less than or equal 50%:";
        int x;
        cin >> x;
        cout << "Splitting dataset (" << 100 - x << "% train," << x << "% test)...\n";
        double splitting = 1.0 * x / 100.0;
        split_dataset(all_data, train_set, test_set, splitting);

        // Train model
        cout << "\nTraining model...\n";
        auto train_start = chrono::high_resolution_clock::now();
        HMM model = train_hmm(train_set);
        auto train_end = chrono::high_resolution_clock::now();
        auto train_duration = chrono::duration_cast<chrono::seconds>(train_end - train_start);
        cout << "Training completed in " << train_duration.count() << " seconds\n";

        // Evaluate model
        cout << "\nEvaluating model on test set...\n";
        evaluate_model(model, test_set);

        // Interactive
        test_model_interactive(model);
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}