#!/usr/bin/env python

import numpy as np
from sklearn import neighbors

def remove_additional_data(kept_numbers, data):
    '''Remove some cases when we test the performace of the program. The original datasets of
    digits contain numbers from 0 to 9. If we want to speed up the test code, we can only
    keep the digits indicated in the variable kept_numbers.    
    '''
    deleted_rows = []
    for i in xrange(data.shape[0]):
        if(data[i][0] in kept_numbers):
            continue
        else:
            deleted_rows.append(i)
    return np.delete(data, deleted_rows, axis=0)

def download_corpus(url):
    import urllib
    print 'downloading \n'+url
    fname = url.split('/')[-1]
    urllib.urlretrieve (url, fname)
    return
                                    
def main():
    download_corpus('http://pjreddie.com/media/files/mnist_train.csv')
    download_corpus('http://pjreddie.com/media/files/mnist_test.csv')
    
    #True: test the best value of K for KNN alg. using k-fold validation
    #False: generate the final result using the best K determined from k-fold validation
    CV_TEST_OF_K = False

    original_train_data = np.genfromtxt('mnist_train.csv', delimiter=',')
    original_test_data = np.genfromtxt('mnist_test.csv', delimiter=',')
    
    # np.save('/home/zhouran/cache/mnist_train.binary', original_train_data)
    # np.save('/home/zhouran/cache/mnist_test.binary', original_test_data)

    # original_train_data = np.load('/home/zhouran/cache/mnist_train.binary.npy')
    # original_test_data = np.load('/home/zhouran/cache/mnist_test.binary.npy')
    
    #setup the digits left in the datasets. Here I use 0-9. If you want
    #the program runs faster, you can use a subset of the full datasets.
    kept_numbers = range(10)
    original_train_data = remove_additional_data(kept_numbers, original_train_data)
    original_test_data = remove_additional_data(kept_numbers, original_test_data)
    print 'The digits in the datasets: ', kept_numbers

    #setup the random numbers seed so that the results are reproduceable
    np.random.seed(1236)
    #shuffle the original data
    np.random.shuffle(original_train_data)
    np.random.shuffle(original_test_data)

    weights = 'uniform'
    #use parallel version of KNN
    n_jobs = 5

    all_train_x = original_train_data[:,1:]
    all_train_y = original_train_data[:,0]

    if(CV_TEST_OF_K):
        #this part uses k-fold validation to find the best value of K in the KNN alg.
        k_fold = KFold(n=len(original_train_data), n_folds=5)
        error_rates = []
        for n_neighbors in xrange(2, 11):
            print 'n_neighbors = ', n_neighbors
            temp_error_rate = 0
            for train_indices, test_indices in k_fold:
                train_x = all_train_x[train_indices]
                train_y = all_train_y[train_indices]
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=n_jobs)
                clf.fit(train_x, train_y)
        
                test_x = all_train_x[test_indices]
                test_y = all_train_y[test_indices]
                pred_y = clf.predict(test_x)
                diff_y = pred_y-test_y
                N_errors = 0
                for diff in diff_y:
                    if(diff!=0):
                        N_errors += 1
                temp_error_rate += float(N_errors)/len(diff_y)*100/len(k_fold)
            error_rates.append([n_neighbors, temp_error_rate])

        fout = open('error_CV.dat', 'w')
        for n_neighbors, error_rate in error_rates:
            s = '%d %.4f\n' % (n_neighbors, error_rate)
            fout.write(s)
        fout.close()
    else:
        #This is for the production run. The value of K in KNN is selected as 5 from the k-fold
        #validation test above.
        n_neighbors = 5
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=n_jobs)
        clf.fit(all_train_x, all_train_y)
        test_x = original_test_data[:,1:]
        test_y = original_test_data[:,0]    
        pred_y = clf.predict(test_x)
        diff_y = pred_y-test_y
        N_errors = 0
        for diff in diff_y:
            if(diff!=0):
                N_errors += 1
        error_rate = float(N_errors)/len(diff_y)*100
        print '%s %.4f %s' % ('error_rate = ', error_rate, '%')
    


    
main()
