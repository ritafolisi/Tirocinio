import sys
from subprocess import call

# Questo è lo script finale che esegue esperimenti su ogni metodo passato.
# In ingresso: dataset, numero colonna da prendere come label, lista di metodi scritta così: [fknn,svm,...]


#dataset sistemato!

def main ():
    dataset = sys.argv[1]
    col_label = sys.argv[2]
    arg3 = sys.argv[3]
    arg3 = arg3[1:-1]   # per togliere le parentesi
    code = arg3.split(",")
    dict_code = {'fknn' :  {'path' : 'FuzzyKNN', 'name' : 'fknn_script.py'},
    'svm' : {'path' : 'Fuzzy-SVM', 'name' : 'main.py'},
    'fcm' : {'path' : 'Fuzzy-C-master', 'name' : 'fuzzy_c.py'},
    'gfmm' : {'path' : 'fuzzy-min-max-classifier-master', 'name' : 'gfmm_script.py'}}

    for i in code:
        print("Calling " +  i)
        call(["python", dict_code[i]['path']+"/"+dict_code[i]['name'], dataset])
        print('\n')


if __name__ == "__main__":
    main()
