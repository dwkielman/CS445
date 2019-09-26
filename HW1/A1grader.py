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

assignmentNumber = '1'

if False:
    runningInNotebook = False
    print('========================RUNNING INSTRUCTOR''S SOLUTION!')
    # import A2mysolution as useThisCode
    # train = useThisCode.train
    # trainSGD = useThisCode.trainSGD
    # use = useThisCode.use
    # rmse = useThisCode.rmse
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

for func in ['rmse', 'train_linear', 'use_linear', 'train_tanh', 'use_tanh']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')


            
print('''\nTesting
  X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
  T = (((X - 5) * 0.05 +  (X * 0.2) **5) / 5.0 - 5.5) / 6
  w, errors = train_linear(X, T, 0.01, 1000)''')

X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
T = (((X - 5) * 0.05 +  (X * 0.2) **5) / 5.0 - 5.5) / 6


try:
    w, errors = train_linear(X, T, 0.01, 1000)

    correct_w = np.array([[-1.63625495], [ 0.2378017 ]])
    if np.sum(np.abs(w - correct_w)) < 0.1 and len(errors) == 1000:
        g += 15
        print('\n--- 15/15 points. Returned correct values.')
    else:
        print('\n---  0/15 points. Returned incorrect values. w should be {} and len(errors) should be 1000.'.format(correct_w))
except Exception as ex:
    print('\n--- 0/15 points. train_linear raised the exception\n {}'.format(ex))



print('''\nTesting
  prediction = use_linear(w, X)''')

try:
    prediction = use_linear(correct_w, X)
    correct_prediction = np.array([[-1.39845325],
                                   [-1.16065155],
                                   [-0.92284985],
                                   [-0.68504815],
                                   [-0.44724645],
                                   [ 0.26615865],
                                   [ 0.50396036],
                                   [ 0.97956376]])

    if np.mean(np.abs(prediction - correct_prediction)) < 0.1:
        g += 15
        print('\n--- 15/15 points. Returned correct values.')
    else:
        print('\n---  0/15 points. Returned incorrect values. prediction should be {}.'.format(correct_prediction))
except Exception as ex:
    print('\n--- 0/15 points. use_linear raised the exception\n {}'.format(ex))
    


print('''\nTesting
  rmse_error = rmse(prediction, T)''')

try:
    err = rmse(correct_prediction, T)
    correct_err = 0.4811335390142959

    if np.abs(err - correct_err) < 0.05:
        g += 15
        print('\n--- 15/15 points. Returned correct values.')
    else:
        print('\n---  0/15 points. Returned incorrect values. rmse_error should be {}.'.format(correct_err))
except Exception as ex:
    print('\n--- 0/15 points. use_linear raised the exception\n {}'.format(ex))

    

print('''\nTesting
  w, errors = train_tanh(X, T, 0.01, 1000)''')

try:
    w, errors = train_tanh(X, T, 0.01, 1000)

    correct_w = np.array([[-3.31966245], [ 0.37837943]])


    if np.sum(np.abs(w - correct_w)) < 0.1 and len(errors) == 1000:
        g += 15
        print('\n--- 15/15 points. Returned correct values.')
    else:
        print('\n---  0/15 points. Returned incorrect values. w should be {} and len(errors) should be 1000.'.format(correct_w))
except Exception as ex:
    print('\n--- 0/15 points. train_tanh raised the exception\n {}'.format(ex))



print('''\nTesting
  prediction = use_tanh(w, X)''')

try:
    prediction = use_tanh(correct_w, X)
    correct_prediction = np.array([[-0.99444025],
                                   [-0.98818734],
                                   [-0.97499014],
                                   [-0.94743866],
                                   [-0.89120773],
                                   [-0.28455094],
                                   [ 0.08554281],
                                   [ 0.68713689]])

    if np.mean(np.abs(prediction - correct_prediction)) < 0.1:
        g += 15
        print('\n--- 15/15 points. Returned correct values.')
    else:
        print('\n---  0/15 points. Returned incorrect values. prediction should be {}.'.format(correct_prediction))
except Exception as ex:
    print('\n--- 0/15 points. use_tanh raised the exception\n {}'.format(ex))
    


name = os.getcwd().split('/')[-1]

print('\n{} Execution Grade is {} / 75'.format(name, g))

print('\n Remaining 25 points will be based on your text describing the derivation of gradients, explanations of code, and plots.')

print('\n{} FINAL GRADE is   / 100'.format(name))

print('\n{} EXTRA CREDIT is   / 1'.format(name))

