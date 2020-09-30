import sys
from subprocess import call
import logging

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
    'fcm' : {'path' : 'Fuzzy-C', 'name' : 'fuzzy_c.py'},
    'gfmm' : {'path' : 'GFMM', 'name' : 'gfmm_script.py'}}

    for i in code:
        print("Calling " +  i)
        logging.basicConfig(filename = 'esperimenti.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
        logging.info('Questo esperimento lavora con il metodo: %s', i)
        call(["python", dict_code[i]['path']+"/"+dict_code[i]['name'], dataset])
        print('\n')
        logging.info('\nProssimo esperimento\n')


if __name__ == "__main__":
    main()
