import sys
import string
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
abbreviations = {"Mr", "Mrs", "Dr", "Prof", "St", "vs", "e.g", "i.e", "etc", "U.S", "Co", 'Inn'}

def define_features(data, idx):
    L = data[idx][1][:-1] if idx > 0 else ""  
    R = data[idx + 1][1] if idx < len(data) - 1 else ""  
    is_single_punctuation = int(len(L) == 1 and L in string.punctuation)
    features = [
        L,  
        R,  
        int(len(L) < 4),  
        int(L.isdigit()),  
        int(R.istitle()),  
        int(L.istitle()),
        1 if L in abbreviations else 0, 
        is_single_punctuation,
    ]
    
    return features
def extract_features(file_path):
    with open(file_path, 'r') as f:
        data = [l.strip().split() for l in f.readlines()]
    
    feature_vectors = []
    labels = []
    valid_labels = {'EOS', 'NEOS'} 

    for idx, row in enumerate(data):
        if len(row) < 3:
            continue
        
        serial, token, label = row
        
        if token.endswith('.'):
            if label in valid_labels:
                features = define_features(data, idx)
                feature_vectors.append(features)
                labels.append(1 if label == 'EOS' else 0)
            else:
                print(f"Label is not valid '{label}' at line {idx+1}, we are skipping.")
    
    return feature_vectors, labels

def fn_calculate_accurac(L_test, predictions):
    count = 0
    n = len(L_test)
    for i in range(n):
        if L_test[i] == predictions[i]:
            count += 1
    return count / n

def main(SBD_train_data, SBD_test_data):
    print("Inside the main")
    F_trained, L_trained = extract_features(SBD_train_data)
    L_words = [f[0] for f in F_trained]
    R_words = [f[1] for f in F_trained]
    L_num = [hash(word) % (10**6) for word in L_words]  
    R_num = [hash(word) % (10**6) for word in R_words]
    for i, feature in enumerate(F_trained):
        F_trained[i][0] = L_num[i]
        F_trained[i][1] = R_num[i]
    clf = DecisionTreeClassifier()
    clf.fit(F_trained, L_trained)
    

    F_test, L_test = extract_features(SBD_test_data)
    L_words_test = [f[0] for f in F_test]
    R_words_test = [f[1] for f in F_test]
    L_W_test = [hash(word) % (10**6) for word in L_words_test]
    R_num_test = [hash(word) % (10**6) for word in R_words_test]
    for i, feature in enumerate(F_test):
        F_test[i][0] = L_W_test[i]
        F_test[i][1] = R_num_test[i]
    
    predictions = clf.predict(F_test)
    
    accuracy = fn_calculate_accurac(L_test, predictions)
    print(f"Accuracy: {accuracy * 100:.3f}%")

    feature_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
    
    tree_rules = export_text(clf, feature_names=feature_names)
    print(classification_report(L_test, predictions))
    print("--------------------------------------")
    print(confusion_matrix(L_test, predictions))
    with open(SBD_test_data, 'r') as f:
        test_data = [line.strip().split() for line in f.readlines()]
    
    output = []
    pred_idx = 0
    
    for idx, row in enumerate(test_data):
        if len(row) < 3:
            continue
        serial, token, label = row
        if token == '.':
            pred_label = 'EOS' if predictions[pred_idx] == 1 else 'NEOS'
            output.append(f"{serial} {token} {pred_label}")
            pred_idx += 1
        else:
            output.append(f"{serial} {token} {label}")


    with open("SBD.test.out", "w") as f:
        f.write("\n".join(output))

if __name__ == '__main__':
    print("Start of the program")
    while len(sys.argv) != 3:
        print("Please check the number of arguments, expecting-'pythonSBD.py SBD.train SBD.test'")
        break  
    SBD_train_data = sys.argv[1]
    SBD_test_data = sys.argv[2]  
    main(SBD_train_data, SBD_test_data)