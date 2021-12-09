from numpy import linalg as LA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
import time
from scipy.sparse.linalg import eigs
from numpy.linalg import inv
import sys


class PFSA:
    def __init__(self, n_partitioning, depth,
                 normalization_method='standardization',
                 train_regime='class wise',
                 matrix_init_regime='MAP',
                 classifier='projection 2'):
        """
        USAGE:
            model = PFSA(n_partitioning, depth, normalize-regime, matrix_init_regime)

        INPUT:
            n_partitioning        - number of partitioning cells [integer]

            depth                 - depth of state [integer]

            normalization_method  - 'standardization' (default) or 'normalization'

            train_regime          - 'instance wise 1', 'instance wise 2', 'instance wise 3' 'class wise' or 'mixed classes'

                                    'instance wise 1': normalize instance wise and get partitioning bounds class wise
                                    'instance wise 2': normalize instance wise and get same partitioning bounds for all classes
                          (default) 'instance wise 3': normalize instance wise and get partitioning bounds instance wise
                                    'class wise': normalize class wise and get partitionging bounds class wise
                                    'mixed classes': normalize based on all classes together and get same partitioning bounds for all classes 

            matrix_init_regime    - 'MAP' or 'ML'

                          (default) 'MAP': initialize morph matrix with all 1
                                    'ML': initialize morph matrix with all 0

            classifier            - 'min norm', 'parity space', 'projection 1', 'projection 2'

                                    'min norm': argmin(norm difference)
                                    'parity space': argmax(<Chi,state probability vector>)
                                    'projection 1': argmin(norm of state prob vector projected on right eigen vectors space)
                          (default) 'projection 2': argmin(norm of state prob vector projected on left null space) 

        Reminder:
            classifier can be modified at the stage of prediction

        OUTPUT:
            a PFSA model
        """

        self.partitioning_number = n_partitioning
        self.depth = depth
        self.normalization_method = normalization_method
        self.train_regime = train_regime
        self.morph_regime = matrix_init_regime
        self.classifier = classifier

        self.partitioning_bounds = {}
        self.feature_number = None
        self.normalize_x_min = {}
        self.normalize_x_max = {}
        self.morph_matrix = {}
        self.morph_count_matrix = {}
        self.alphabet_size = None
        self.states_size = None
        self.morph_regime = 'MAP'
        self.standard_std = {}
        self.standard_mean = {}
        self.classes = None  # a list of classes
        self.state_pro_v = {}
        self.right_vecs_matrix = {}
        self.distances_cl = None

        self.P_rv = {}  # the projection matrix to each class's right eigenvector space
        # the projection matrix to each class's left null space.
        self.P_ln = {}
        self.Chi = {}

    # @numba.jit(boundscheck=True)
    def fit(self, x, y):
        """
        USAGE:
            model.fit(x,y)

        INPUT:
            x - training x, 3D array [samples, time steps, features]
            y - training labels 1D array
            normalize
        """
        x = x.copy()
        (x_l, x_m, x_n) = x.shape

        n_partitioning = self.partitioning_number
        depth = self.depth
        matrix_init_regime = self.morph_regime
        self.feature_number = x_n
        alphabet_size = n_partitioning ** x_n
        states_size = alphabet_size ** depth
        classifier = self.classifier
        self.alphabet_size = alphabet_size
        self.states_size = states_size

        classes = list(np.unique(y))
        self.classes = classes
        train_regime = self.train_regime
        normalization_method = self.normalization_method

        if train_regime == 'mixed classes':
            # ..................... normalization................
            print("normalization for all classes together")
            (a, b, X_train_normalized) = self.normalize(x, normalization_method)
            if normalization_method == 'normalization':
                for class_i in classes:
                    self.normalize_x_min[class_i] = a
                    self.normalize_x_max[class_i] = b
            elif normalization_method == 'standardization':
                for class_i in classes:
                    self.standard_std[class_i] = a
                    self.standard_mean[class_i] = b
            else:
                for class_i in classes:
                    self.standard_std[class_i] = a
                    self.standard_mean[class_i] = b
            # ..................... get bounds................
            print('get the same bounds for all classes')
            X_train_bounds = self.determine_partitioning_bounds(
                X_train_normalized, n_partitioning)
            for class_i in classes:
                self.partitioning_bounds[class_i] = X_train_bounds
            # ..................... get symbols...............
            print('get symbols for all classes')
            X_train_symbols = self.generate_symbols(
                X_train_normalized, X_train_bounds)
            # ..................... get states...............
            print('get states for all classes')
            X_train_states = self.generate_states(
                X_train_symbols, depth, alphabet_size)
            for class_i in classes:
                # ..................... get morph matrix...............
                print('get morph matrix %f' % class_i)
                X_train_states_class_i = X_train_states[np.where(y == class_i)]
                (morph_count_matrix, morph_matrix) = self.cal_morph_matrix(
                    X_train_states_class_i, states_size, matrix_init_regime)
                self.morph_count_matrix[class_i] = morph_count_matrix
                self.morph_matrix[class_i] = morph_matrix

        elif train_regime == 'class wise':
            for class_i in classes:
                # ..................... normalization................
                print("normalization for class %f" % class_i)
                X_train = x[np.where(y == class_i)]
                (a, b, X_train_normalized) = self.normalize(
                    X_train, normalization_method)
                if normalization_method == 'normalization':
                    self.normalize_x_min[class_i] = a
                    self.normalize_x_max[class_i] = b
                elif normalization_method == 'standardization':
                    self.standard_std[class_i] = a
                    self.standard_mean[class_i] = b
                else:
                    print('bug')
                    self.standard_std[class_i] = a
                    self.standard_mean[class_i] = b
                # ..................... get bounds................
                print('get bounds of class %f' % class_i)
                X_train_bounds = self.determine_partitioning_bounds(
                    X_train_normalized, n_partitioning)
                self.partitioning_bounds[class_i] = X_train_bounds
                # ..................... get symbols...............
                print('get symbols of class %f' % class_i)
                X_train_symbols = self.generate_symbols(
                    X_train_normalized, X_train_bounds)
                # ..................... get states...............
                print('get states of class %f' % class_i)
                X_train_states = self.generate_states(
                    X_train_symbols, depth, alphabet_size)
                # ..................... get morph matrix...............
                print('get morph matrix %f' % class_i)
                (morph_count_matrix, morph_matrix) = self.cal_morph_matrix(
                    X_train_states, states_size, matrix_init_regime)
                self.morph_count_matrix[class_i] = morph_count_matrix
                self.morph_matrix[class_i] = morph_matrix

        elif train_regime == 'instance wise 1':
            # ..................... normalization................
            print("normalization for each sample")
            X_train_normalized = self.normalize2(x, normalization_method)

            for class_i in classes:
                X_train_normalized_class_i = X_train_normalized[np.where(
                    y == class_i)]
                # ..................... get bounds................
                print('get bounds of class %f' % class_i)
                X_train_bounds = self.determine_partitioning_bounds(
                    X_train_normalized_class_i, n_partitioning)
                self.partitioning_bounds[class_i] = X_train_bounds
                # ..................... get symbols...............
                print('get symbols of class %f' % class_i)
                X_train_symbols = self.generate_symbols(
                    X_train_normalized_class_i, X_train_bounds)
                # ..................... get states...............
                print('get states of class %f' % class_i)
                X_train_states = self.generate_states(
                    X_train_symbols, depth, alphabet_size)
                # ..................... get morph matrix...............
                print('get morph matrix %f' % class_i)
                (morph_count_matrix, morph_matrix) = self.cal_morph_matrix(
                    X_train_states, states_size, matrix_init_regime)
                self.morph_count_matrix[class_i] = morph_count_matrix
                self.morph_matrix[class_i] = morph_matrix

        elif train_regime == 'instance wise 2':
            # ..................... normalization................
            print("normalization for each sample")
            X_train_normalized = self.normalize2(x, normalization_method)
            # ..................... get bounds................
            print('get the same bounds for all classes')
            X_train_bounds = self.determine_partitioning_bounds(
                X_train_normalized, n_partitioning)
            for class_i in classes:  # store the partitioning bounds
                self.partitioning_bounds[class_i] = X_train_bounds
            # ..................... get symbols...............
            print('get symbols for all classes')
            X_train_symbols = self.generate_symbols(
                X_train_normalized, X_train_bounds)
            # ..................... get states...............
            print('get states for all classes')
            X_train_states = self.generate_states(
                X_train_symbols, depth, alphabet_size)
            for class_i in classes:
                X_train_states_class_i = X_train_states[np.where(y == class_i)]
                # ..................... get morph matrix...............
                print('get morph matrix %f' % class_i)
                (morph_count_matrix, morph_matrix) = self.cal_morph_matrix(
                    X_train_states_class_i, states_size, matrix_init_regime)
                self.morph_count_matrix[class_i] = morph_count_matrix
                self.morph_matrix[class_i] = morph_matrix

        else:
            if train_regime == 'instance wise 3':
                pass
            else:
                self.train_regime = 'instance wise 3'
                print("unknown train regime, using instance wise 3")
            # ..................... normalization................
            print("normalization for each sample")
            X_train_normalized = self.normalize2(x, normalization_method)
            # ..................... get bounds and symbols................
            print('get the partitioning bounds and generate symbols instance wise')
            X_train_symbols = self.determine_bounds_and_generate_symbols(
                X_train_normalized, n_partitioning)
            # ..................... get states...............
            print('get states for all classes')
            X_train_states = self.generate_states(
                X_train_symbols, depth, alphabet_size)
            for class_i in classes:
                X_train_states_class_i = X_train_states[np.where(y == class_i)]
                # ..................... get morph matrix...............
                print('get morph matrix %f' % class_i)
                (morph_count_matrix, morph_matrix) = self.cal_morph_matrix(
                    X_train_states_class_i, states_size, matrix_init_regime)
                self.morph_count_matrix[class_i] = morph_count_matrix
                self.morph_matrix[class_i] = morph_matrix

        for class_i in classes:
            print('class %f, calculate the left eigenvector corresponding to left eigenvalue 1, this is the state probability vector.' % class_i)
            # ......... calculate the left eigenvector corresponding to eigen value 1....................
            try:
                vals_1, vecs_1 = eigs(
                    self.morph_matrix[class_i].T, k=1, sigma=1)
            except:
                vals_1, vecs_1 = eigs(self.morph_matrix[class_i].T, k=1)
            # cautrion: here the state probability vector is in column form
            self.state_pro_v[class_i] = np.real(vecs_1)
            print('class %f, calculate the right eigenvectors corresponding to right eigenvalue (excludes eigen value 1).' % class_i)
            # .......... calculate the right eigenvectors corresponding to right eigenvalue (excludes eigen value 1)......
            vals, vecs = eigs(self.morph_matrix[class_i])
            # note: the right eigen vectors are the columns of this matrix
            right_vecs_matrix = vecs[:, 1:]
            self.right_vecs_matrix[class_i] = right_vecs_matrix

        for class_i in classes:
            print('class %f, calculate state weight Chi' % class_i)
            self._cal_state_weight()
            print('class %f, calculate projection matrix 1' % class_i)
            vecs_1 = self.state_pro_v[class_i]
            self.P_ln[class_i] = np.eye(
                states_size) - vecs_1.dot(inv(vecs_1.T.dot(vecs_1))).dot(vecs_1.T)
            print('class %f, calculate projection matrix 2' % class_i)
            right_vecs_matrix = self.right_vecs_matrix[class_i]
            self.P_rv[class_i] = right_vecs_matrix.dot(
                inv(right_vecs_matrix.T.dot(right_vecs_matrix))).dot(right_vecs_matrix.T)

    # @numba.jit(boundscheck=True)
    def predict(self, x, classifier=None):
        x = x.copy()
        (x_l, x_m, x_n) = x.shape
        classes = self.classes
        classes_number = len(classes)
        morph_matrix = self.morph_matrix
        depth = self.depth
        states_size = self.states_size
        alphabet_size = self.alphabet_size
        morph_regime = self.morph_regime
        normalization_method = self.normalization_method
        x_min = self.normalize_x_min
        x_max = self.normalize_x_max
        x_std = self.standard_std
        x_mean = self.standard_mean
        partitioning_bounds = self.partitioning_bounds
        train_regime = self.train_regime

        if classifier == None:
            classifier = self.classifier
            
        num_classifier = len(classifier)

        # ................ normalization.................
        if train_regime in ['instance wise 1', 'instance wise 2', 'instance wise 3']:
            print('instance wise normalization')
            X_normalized = self.normalize2(x, normalization_method)

        distances = np.zeros((x_l, classes_number))
        distances_cl = {}
        for cl in classifier:
            distances_cl[cl] = distances.copy()
            
        for i in range(classes_number):
            class_i = classes[i]
            print('class %i' % class_i)
            # ............. normalization..................
            if train_regime not in ['instance wise 1', 'instance wise 2', 'instance wise 3']:
                print('normalization')
                # (x- x_min) / (x_max - x_min)
                if normalization_method == 'normalization':
                    X_normalized = self.normalize_continue(
                        x, normalization_method, x_min=x_min[class_i], x_max=x_max[class_i])
                # (x - x_mean) / (x_std)
                elif normalization_method == 'standardization':
                    X_normalized = self.normalize_continue(
                        x, normalization_method, x_std=x_std[class_i], x_mean=x_mean[class_i])
            # ............. generate symbols...............
            if train_regime not in ['instance wise 3']:
                X_bounds = partitioning_bounds[class_i]
                X_symbols = self.generate_symbols(X_normalized, X_bounds)
            elif train_regime in ['instance wise 3']:
                X_symbols = self.determine_bounds_and_generate_symbols(
                    X_normalized, n_partitioning)
            else:
                pass

            # ............. generate states...............
            print('get states for all classes')
            X_states = self.generate_states(X_symbols, depth, alphabet_size)
            # ............. generate morph matrix...............
            print('get morph matrix')
            for l in range(x_l):
                (morph_count_matrix, morph_matrix_l) = self.cal_morph_matrix(
                    X_states[l:l+1], states_size, morph_regime)
                # .............. calculate state probability vector............
                try:
                    val_1, vec_1 = eigs(morph_matrix_l.T, k=1, sigma=1)
                except:
                    val_1, vec_1 = eigs(morph_matrix_l.T, k=1)
                vec_1 = np.real(vec_1)
                if 'min norm' in classifier:
                    distance_l = LA.norm(
                        morph_matrix[class_i] - morph_matrix_l)
                    distances_cl['min norm'][l, i] = distance_l
                if 'parity space' in classifier:
                    distance_l = - abs(self.Chi[class_i].dot(vec_1))
                    distances_cl['parity space'][l, i] = distance_l
                if 'projection 1' in classifier:
                    distance_l = LA.norm(self.P_rv[class_i].dot(vec_1))
                    distances_cl['projection 1'][l, i] = distance_l
                if 'projection 2' in classifier:
                    distance_l = LA.norm(self.P_ln[class_i].dot(vec_1))
                    distances_cl['projection 2'][l, i] = distance_l
                if ('min norm' not in classifier) and \
                ('parity space' not in classifier) and \
                ('projection 1' not in classifier) and \
                ('projection 2' not in classifier):
                    print('no valid classifier')
                    sys.exit()
                #distances[l, i] = distance_l
        print('get labels')
        Y_labels = np.zeros((x_l, num_classifier))
        self.distances_cl = distances_cl
        for i in range(num_classifier):
            cl = classifier[i]
            distances = distances_cl[cl]
            max_index_row = list(np.argmin(distances, axis=1))
            for j in range(x_l):
                max_index_row[j] = classes[max_index_row[j]]
            Y_labels[:, i] = max_index_row
            
        if num_classifier == 1:
            return Y_labels[:,0]
        else: return Y_labels

    def normalize(self, x, method='normalization'):
        # c is a list of class
        # 0, 1 normalization or mean std standardization
        x = x.copy()
        (x_l, x_m, x_n) = x.shape
        x = x.reshape(x_l * x_m, x_n)
        if method == 'normalization':
            x_max = np.amax(x, axis=0)
            x_min = np.amin(x, axis=0)
            x = (x - x_min) / (x_max - x_min)
            x = x.reshape(x_l, x_m, x_n)
            return x_min, x_max, x
        else:
            if method == 'standardization':
                pass
            else:
                print('bug')
            x_std = np.std(x, axis=0)
            x_mean = np.mean(x, axis=0)
            x = (x - x_mean) / x_std
            x = x.reshape(x_l, x_m, x_n)
            return x_std, x_mean, x

    @numba.jit(boundscheck=True)
    def normalize2(self, x, method='normalization'):
        x = x.copy()
        (x_l, x_m, x_n) = x.shape
        if method == 'normalization':
            for l in range(x_l):
                x_max_l = np.amax(x[l], axis=0)
                x_min_l = np.amin(x[l], axis=0)
                x[l] = (x[l] - x_min_l) / (x_max_l - x_min_l)
        if method == 'standardization':
            for l in range(x_l):
                x_std_l = np.std(x[l], axis=0)
                x_mean_l = np.mean(x[l], axis=0)
                x[l] = (x[l] - x_mean_l) / x_std_l
        return x

    def normalize_continue(self, x, method, x_min=None, x_max=None, x_std=None, x_mean=None):
        x = x.copy()
        if method == 'normalization':
            return (x - x_min) / (x_max - x_min)
        if method == 'standardization':
            return (x - x_mean) / x_std

    @numba.jit(boundscheck=True)
    def determine_partitioning_bounds(self, x, n):
        # maximum entropy partitioning
        # x: 3d np array
        #   - the first dimension contains the instances, there are x_l instances
        #   - the second dimension contains the time sereis steps, there are x_m steps
        #   - the thrid dimension contains the features, there are x_n features
        #
        # n: number of partitioning
        # c: a list of classes
        #
        # return a list of bounds for features; each feature has a list of bounds

        x = x.copy()
        (x_l, x_m, x_n) = x.shape
        x = x.reshape(x_l * x_m, x_n)
        (x_m, x_n) = x.shape
        (Ntotal, Nsections) = (x_m, n)
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section+1] +
                         (Nsections-extras) * [Neach_section])
        div_points = np.array(section_sizes, dtype=np.intp).cumsum()
        div_points = div_points[1:-1] - 1
        bounds = []
        for i in range(0, x_n):
            x_i = np.sort(x[:, i])
            bounds_i = [x_i[j] for j in div_points]
            bounds = bounds + [bounds_i]
        return bounds

    #@numba.jit(boundscheck=True)
    def generate_symbols(self, x, bounds):
        # x: 3d np array
        #   - the first dimension contains the instances, there are x_l instances
        #   - the second dimension contains the time sereis steps, there are x_m steps
        #   - the thrid dimension contains the features, there are x_n features
        #
        # bounds: a list of bounds for features; each feature has a list of bounds
        #
        # return an array of symbols corresponding to the steps
        x = x.copy()
        (x_l, x_m, x_n) = x.shape

        # symbols = np.zeros(x_m)
        # n = len(bounds[0]) + 1
        n = self.partitioning_number
        for l in range(0, x_l):
            for i in range(0, x_n):
                # iterate each column (feature)
                x_i = x[l, :, i]
                bounds_i = bounds[i]
                mask_i = np.zeros(x_i.size)
                bounds_i_rev = bounds_i[::-1]
                for k in range(0, len(bounds_i_rev)):
                    mask_i = mask_i + (x[l, :, i] > bounds_i_rev[k])
                x[l, :, i] = mask_i  # .astype(np.int)
        symbols = self._ravel_index(x, np.array([n] * x_n))
        symbols = symbols.astype(np.int)
        return symbols

    def determine_bounds_and_generate_symbols(self, x, n):
        """
        INPUT
            x - training x, 3D array [samples, time steps, features]
            n - number of partitioning cells

        OUTPUT
            x_symbols, 2D array [samples, time steps]
        """

        x = x.copy()
        (x_l, x_m, x_n) = x.shape

        (Ntotal, Nsections) = (x_m, n)
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section+1] +
                         (Nsections-extras) * [Neach_section])
        div_points = np.array(section_sizes, dtype=np.intp).cumsum()
        div_points = div_points[1:-1] - 1

        for l in range(x_l):
            for i in range(0, x_n):
                x_i_s = np.sort(x[l, :, i])
                bounds_i = [x_i_s[j] for j in div_points]

                # iterate each column (feature)
                x_i = x[l, :, i]
                mask_i = np.zeros(x_i.size)
                bounds_i_rev = bounds_i[::-1]
                for k in range(0, len(bounds_i_rev)):
                    mask_i = mask_i + (x[l, :, i] > bounds_i_rev[k])
                x[l, :, i] = mask_i  # .astype(np.int)
        symbols = self._ravel_index(x, np.array([n] * x_n))
        symbols = symbols.astype(np.int)
        return symbols

    @numba.jit(boundscheck=True)
    def generate_states(self, x, depth, alphabet_size):
        # x is symbol sereies
        #   - rows are instances
        #   - columns are steps
        x = x.copy()
        dim = alphabet_size
        if depth == 1:
            return x
        else:
            (x_m, x_n) = x.shape
            states = np.zeros((x_m, x_n - depth + 1))
        (states_m, states_n) = states.shape

        for i in range(0, states_m):
            for j in range(0, states_n):
                r = 0
                for k in range(depth):
                    r = r * dim
                    r = r + x[i, j + k]
                states[i, j] = r
        states = states.astype(np.int)
        return states

    # use numba jit to speed up calculation; jit package has tremendous bugs; it is selectively used
    @numba.jit(boundscheck=True)
    def cal_morph_matrix(self, x, states_size, regime='MAP'):
        # n is number of states
        # x is symbol series
        # x_m is the number of instances
        # x_n is the length of state series data

        # print(series_lens)
        x = x.copy()

        (x_m, x_n) = x.shape

        if regime == 'MAP':
            morph_matrix = np.ones((states_size, states_size))
        if regime == 'ML':
            morph_matrix = np.zeros((states_size, states_size))

        for m in range(x_m):
            for n in range(x_n-1):
                morph_matrix[x[m, n], x[m, n+1]
                             ] = morph_matrix[x[m, n], x[m, n+1]] + 1

        morph_count_matrix = morph_matrix.copy()
        for i in range(0, states_size):
            morph_matrix[i] = morph_matrix[i] / morph_matrix[i].sum()
        morph_matrix[np.isnan(morph_matrix)] = 0
        return (morph_count_matrix, morph_matrix)

    def _cal_state_weight(self):
        # get H_i
        state_num = self.states_size
        class_list = self.classes
        class_num = len(class_list)
        H = np.zeros((state_num, class_num))
        for i in range(class_num):
            H[:, i:i+1] = self.state_pro_v[class_list[i]]
        for i in range(class_num):
            class_i = class_list[i]
            H_i = np.delete(H, i, 1)
            Chi = self.state_pro_v[class_i].T.dot(
                np.eye(state_num) - H_i.dot(np.linalg.inv(H_i.T.dot(H_i))).dot(H_i.T))
            self.Chi[class_i] = Chi

    def probability_generating_sequence(self, sequence):
        # todo
        w, v = LA_sci.eig(self.morph_matrix, left=True, right=False)
        eigv1_left = v[:, np.where(w == 1)[0][0]]
        eigv1_left = eigv1_left / eigv1_left.sum()
        eigv1_left[sequence[0]]
        for i in range(0, len(sequence) - 1):
            p = p * morph_matrix[sequence[i], sequence[i + 1]]
        return p

    # use numba jit to speed up calculation; jit package has tremendous bugs; it is selectively used
    @numba.jit(boundscheck=True)
    def _ravel_index(self, x, dims):
        (x_l, x_m, x_n) = x.shape

        symbols = np.zeros((x_l, x_m))
        n_dim = dims.shape[0]
        # rows of x is the coordinate, for example: (0,0,1)
        # dims is the slices in each dimension, for example: (2, 2, 2)
        # output is the index of the hypercube, in the above example case, return 1
        for l in range(0, x_l):
            for m in range(0, x_m):
                i = 0
                for dim, j in zip(dims, x[l, m]):
                    i = i * dim
                    i = i + j
                symbols[l, m] = i
        return symbols

    # use numba jit to speed up calculation; jit package has tremendous bugs; it is selectively used
    @numba.jit(boundscheck=True)
    def _ravel_state(self, x, dim):
        # x is symbol sereies
        #   - rows are instances
        #   - columns are steps
        i = 0
        x_list = list(x)
        for j in x_list:
            i = i * dim
            i = i + j
        return i

    def save_pfsa(self):
        # todo
        pass

    def load_pfsa(self):
        # todo
        pass
