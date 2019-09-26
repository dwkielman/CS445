import os
import copy
import signal

# Code to limit running time of specific parts of code.
#  To use, do this for example...
#
#  signal.alarm(seconds)
#  try:
#    ... run this ...
#  except TimeoutException:
#     print(' 0/8 points. Your depthFirstSearch did not terminate in', seconds/60, 'minutes.')
# Exception to signal exceeding time limit.


# class TimeoutException(Exception):
#     def __init__(self, *args, **kwargs):
#         Exception.__init__(self, *args, **kwargs)


# def timeout(signum, frame):
#     raise TimeoutException

# seconds = 60 * 5

# Comment next line for Python2
# signal.signal(signal.SIGALRM, timeout)

import os
import numpy as np

print('\n======================= Code Execution =======================\n')

assignmentNumber = '2'

if False:
    runningInNotebook = False
    print('========================RUNNING INSTRUCTOR''S SOLUTION!')
else:
    import subprocess, glob, pathlib
    filename = next(glob.iglob('*-A{}.ipynb'.format(assignmentNumber)), None)
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         '*-A{}.ipynb'.format(assignmentNumber), '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import)):  #  and
            # not isinstance(node, ast.ClassDef) and
            # not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *



g = 0

for func in ['run_parameters']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')


print('''Testing:
data = np.loadtxt('machine.data', delimiter=',', usecols=range(2, 10))
X = data[:, :-2]
T = data[:, -2:-1]
Xtrain = X[:160, :]
Ttrain = T[:160, :]
Xtest = X[160:, :]
Ttest = T[160:, :]

means = Xtrain.mean(0)
stds = Xtrain.std(0)
Xtrains = (Xtrain - means) / stds
Xtests = (Xtest - means) / stds

results = run_parameters(Xtrains, Ttrain, Xtests, Ttest, [10, 100], [0.001, 0.01], [5, 10], [1, 50], verbose=False)''')
      
data = np.loadtxt('machine.data', delimiter=',', usecols=range(2, 10))
X = data[:, :-2]
T = data[:, -2:-1]
Xtrain = X[:160, :]
Ttrain = T[:160, :]
Xtest = X[160:, :]
Ttest = T[160:, :]

means = Xtrain.mean(0)
stds = Xtrain.std(0)
Xtrains = (Xtrain - means) / stds
Xtests = (Xtest - means) / stds

try:
    results = run_parameters(Xtrains, Ttrain, Xtests, Ttest, [10, 100], [0.001, 0.01], [5, 10], [1, 50], verbose=False)
    print(results)
    print()
    g += 40
    print('\n--- 40/40 points. run_parameters terminates.')
except Exception as ex:
    print('\n--- 0/40 points. run_parameters raises exception', ex)


if results.shape == (32, 7):
    g += 20
    print('\n--- 20/20 points. results.shape is correct.')
else:
    print('\n--- 0/20 points. results.shape should be (32, 7).  Yours is {}'.format(results.shape))
    

correct_columns = ['Algorithm', 'Epochs', 'Learning Rate', 'Hidden Units', 'Batch Size', 'RMSE Train', 'RMSE Test']
if results.columns.tolist() == correct_columns:
    g += 10
    print('\n--- 10/10 points. results column names is correct.')
else:
    print('\n--- 0/10 points. results column names is incorrect.\n   Should be {}'.format(correct_columns))

algo = results['Algorithm'].iloc[0].lower()
if algo in ['adam', 'sgd']:
    g += 10
    print('\n--- 10/10 points. Values in results Algorithm column correct.')
else:
    print('\n--- 0/10 points. Value in results Algorithm column of {} is incorrect.'.format(algo))
    
top_epoch = results.sort_values('RMSE Test').iloc[0]['Epochs']
if top_epoch  == 100:
    g += 10
    print('\n--- 10/10 points. Number of epochs for lowest RMSE Train is correct.')
else:
    print('\n--- 0/10 points. Number of epochs for lowest RMSE Train is {}.  It should be 100.'.format(top_epoch))
    
    



name = os.getcwd().split('/')[-1]

print('\n{} Execution Grade is {} / 90'.format(name, g))

print('\n Remaining 10 points will be based on your text descriptions of results and plots.')

print('\n{} FINAL GRADE is   / 100'.format(name))

print('\n{} EXTRA CREDIT is   / 1'.format(name))

